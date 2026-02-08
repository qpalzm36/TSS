import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
from tqdm import tqdm
import torch
import torch.multiprocessing as mp 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vllm import LLM, SamplingParams
import argparse  # <--- 新增导入
MAX_NEW_TOKENS = 512
MAX_STEPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def protect_math_expressions(content: str):
    math_map = {}
    protected_content = content
    math_patterns = [
        (r'\\\\?frac\{[^}]*\}\{[^}]*\}', 'FRAC'), (r'\\\\?sqrt\{[^}]*\}', 'SQRT'),
        (r'\\\\?boxed\{[^}]*\}', 'BOXED'), (r'\\\\?text\{[^}]*\}', 'TEXT'),
        (r'\\\\?left\([^)]*\\\\?right\)', 'PAREN'), (r'\$[^$]*\$', 'DOLLAR'),
    ]
    placeholder_template = "__MATH_EXPR_{}__"
    for pattern, _ in math_patterns:
        matches = list(re.finditer(pattern, protected_content))
        for match in reversed(matches):
            placeholder = placeholder_template.format(len(math_map))
            math_map[placeholder] = match.group(0)
            protected_content = protected_content[:match.start()] + placeholder + protected_content[match.end():]
    return protected_content, math_map

def restore_math_expressions(obj, math_map):
    if isinstance(obj, dict):
        return {k: restore_math_expressions(v, math_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_math_expressions(item, math_map) for item in obj]
    elif isinstance(obj, str):
        result = obj
        for _ in range(3):
            for placeholder, expr in math_map.items():
                if placeholder in result:
                    result = result.replace(placeholder, expr)
        return result
    else:
        return obj

def escape_latex_for_json(s: str) -> str:
    return re.sub(r'(?<!\\)\\([a-zA-Z]+)', r'\\\\\1', s)

def parse_json_robust(raw_content: str, problem_id: int) -> dict:
    if not isinstance(raw_content, str):
        return {"error": "parsing_failed: input is not a string", "original_content": raw_content or ""}
    content = raw_content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if "Invalid \\escape" not in str(e):
            try:
                last_brace_index = content.rfind('}')
                if last_brace_index != -1:
                    return json.loads(content[:last_brace_index + 1])
            except json.JSONDecodeError:
                pass
            return {"error": f"parsing_failed: {str(e)}", "original_content": raw_content}
    try:
        content_escaped = escape_latex_for_json(content)
        return json.loads(content_escaped)
    except json.JSONDecodeError as e_escaped:
        return {"error": f"parsing_failed_after_escape: {str(e_escaped)}", "original_content": raw_content}

def _extract_answer_from_text(text: str) -> str:
    """
    从GSM8K格式的文本中提取最终答案。
    该版本融合了多种策略，按可靠性降序排列。
    """
    if not text:
        return None
    conclusion_match = re.search(r'\[CONCLUSION\](.*)', text, re.DOTALL)
    if conclusion_match:
        # 我们直接返回结论的全部内容，保留所有上下文
        return conclusion_match.group(1).strip().strip('"')
    # 策略 1: 查找 \boxed{} (最高优先级)
    # 使用了更强大的正则表达式，可以处理嵌套括号
    all_boxed_contents = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)

    if not all_boxed_contents:
        return "" # 没找到任何 boxed

    # 2. 定位到最后一个 boxed 的内容
    last_content = all_boxed_contents[-1].strip()

    # 3. 从内容中提取最后一个数字 (增加了对负号 \-? 的支持)
    #    这个正则表达式会匹配 "-1,234.56" 或 "$42" 这样的字符串
    numeric_matches = re.findall(r'\-?[\$\d,]+(?:\.\d+)?', last_content)
    
    if numeric_matches:
        # 4. 清理并返回最后一个找到的数字
        last_number_str = numeric_matches[-1]
        return last_number_str.replace('$', '').replace(',', '').strip()
    
    # 5. 如果里面没数字，返回原始内容
    return last_content

    # 策略 2: 查找 #### 格式 (GSM8K的标准格式，非常可靠的备用选项)
    match = re.search(r'####\s*(.*)$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 策略 3: 查找 "The answer is" 或 "The final answer is" 格式 (通用备用选项)
    match = re.search(r'[Tt]he (?:final )?answer is\s+(.*?)\.?$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # 策略 4 (最后手段): 提取文本中最后一个可能是答案的数字
    # 这个策略非常通用，可以处理模型未按任何特殊格式输出答案的情况
    matches = re.findall(r'[\$\d,]+(?:\.\d+)?', text)
    if matches:
        # 清理并返回最后一个匹配项
        return matches[-1].replace('$', '').replace(',', '').strip()

    # 如果连数字都找不到，返回None
    return text

def extract_final_answer_robust(generated_steps: list) -> str:
    if not generated_steps: return None
    last_step_obj = generated_steps[-1]
    if not isinstance(last_step_obj, dict): return None
    if 'error' in last_step_obj and 'original_content' in last_step_obj:
        return _extract_answer_from_text(last_step_obj['original_content'])
    step_text = ""
    for value in last_step_obj.values():
        if isinstance(value, str):
            step_text = value
            break
    return _extract_answer_from_text(step_text)

# --- 配置 ---
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--generator_model_path", type=str, default="", help="")
    parser.add_argument("--retriever_model_path", type=str, default="", help="")
    parser.add_argument("--knowledge_base_docs_path", type=str, default="", help="")
    parser.add_argument("--faiss_index_path", type=str, default="", help="")
    parser.add_argument("--test_set_path", type=str, default="", help="")
    parser.add_argument("--output_log_path", type=str, default="", help="")

    return parser.parse_args()
args = parse_args()
GENERATOR_MODEL_PATH = args.generator_model_path
RETRIEVER_MODEL_PATH = args.retriever_model_path
KNOWLEDGE_BASE_DOCS_PATH = args.knowledge_base_docs_path
FAISS_INDEX_PATH = args.faiss_index_path
TEST_SET_PATH = args.test_set_path
OUTPUT_LOG_PATH = args.output_log_path



def create_decision_prompt(problem: str, previous_step: str = None) -> str:
    """【Llama-2 版】为决策阶段创建提示"""
    system_instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no retrieval>`."
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {previous_step}")
    user_content = f"{{{', '.join(input_parts)}}}"
    prompt = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_content} [/INST]"
    return prompt

def create_content_prompt(problem: str, previous_step=None, retrieval_context: str = None) -> str:
    """【Llama-2 版】为内容生成阶段创建提示"""
    system_instruction = (
        "You are an expert math solver. Your goal is to generate the next step to solve a math problem. "
        "Your response MUST be a single JSON object with a key like `\"Step N\"` and the explanation as the value. "
        "Use clear, concise language. When you reach the final answer, include it in the CONCLUSION section with \\boxed{answer} format. "
        "If the problem is fully solved, add [END_OF_SOLUTION] *after* the JSON object."
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        if isinstance(previous_step, str):
            try: previous_step_dict = json.loads(previous_step)
            except json.JSONDecodeError: previous_step_dict = None
        else:
            previous_step_dict = previous_step
        if previous_step_dict:
            input_parts.append(f"\"previous_step\": {json.dumps(previous_step_dict)}")
    if retrieval_context:
        retrieval_str = json.dumps(retrieval_context)
        input_parts.append(f"\"retrieval_context\": {retrieval_str}")
    user_content = f"{{{', '.join(input_parts)}}}"
    prompt = f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_content} [/INST]"
    return prompt

def run_retrieval(query, retriever_model, faiss_index, knowledge_base, k=1):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')
    if query_embedding_np.ndim == 1:
        query_embedding_np = np.expand_dims(query_embedding_np, axis=0)
    faiss.normalize_L2(query_embedding_np)
    _, top_k_indices = faiss_index.search(query_embedding_np, k)
    retrieved_docs = [knowledge_base[i] for i in top_k_indices[0]]
    return retrieved_docs[0] if retrieved_docs else None

def extract_answer_from_gsm8k(answer_text):
    """从GSM8K格式的答案中提取最终答案（#### 之后的部分）"""
    match = re.search(r'####\s+(.*?)$', answer_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_inference_vllm():
    try:
        import ray
        if ray.is_initialized(): ray.shutdown()
    except: pass
    try:
        import subprocess
        subprocess.run(["ray", "stop"], capture_output=True, timeout=10)
    except: pass
    
    os.makedirs(os.path.dirname(OUTPUT_LOG_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_LOG_PATH):
        os.remove(OUTPUT_LOG_PATH)

    print("正在加载 vLLM 生成器...")
    llm = LLM(
        model=GENERATOR_MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        dtype="bfloat16",
        enforce_eager=True
    )
    tokenizer = llm.get_tokenizer()

    print(f"正在加载检索器模型: {RETRIEVER_MODEL_PATH}")
    retriever = SentenceTransformer(RETRIEVER_MODEL_PATH, device=DEVICE)
    
    print(f"正在加载知识库: {KNOWLEDGE_BASE_DOCS_PATH}")
    kb_docs = []
    with open(KNOWLEDGE_BASE_DOCS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text_parts = []
            if "Problem" in data: text_parts.append(f"Problem: {data['Problem']}")
            step_keys = sorted([k for k in data.keys() if 'Step' in k], key=lambda k: int(re.search(r'Step (\d+)', k).group(1)))
            for key in step_keys: text_parts.append(f"{key}: {data[key]}")
            kb_docs.append("\n\n".join(text_parts))
    
    print(f"正在加载 FAISS 索引: {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("所有模型和数据加载成功。")
    
    sampling_params_decision = SamplingParams(
        max_tokens=5, temperature=0.0, stop=["<retrieval>", "<no retrieval>"],
        include_stop_str_in_output=True, ignore_eos=True, skip_special_tokens=False
    )
    sampling_params_content = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=0.1, stop=["</s>", "[END_OF_SOLUTION]"],
        include_stop_str_in_output=True
    )

    # 加载GSM8K数据集
    print(f"正在加载GSM8K数据集: {TEST_SET_PATH}")
    test_problems = []
    with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 提取问题和答案
            problem = data["question"]
            answer = extract_answer_from_gsm8k(data["answer"])
            test_problems.append({
                "problem": problem,
                "answer": answer
            })
    print(f"加载了 {len(test_problems)} 个GSM8K问题")

    request_states = [{"problem_id": i, "problem_data": p, "generated_steps": [], "inference_trace": [], "finished": False} for i, p in enumerate(test_problems)]

    for step in range(MAX_STEPS):
        active_requests = [r for r in request_states if not r["finished"]]
        if not active_requests:
            print("所有问题都已解决。")
            break
        print(f"\n--- 第 {step+1} 步, 处理 {len(active_requests)} 个请求 ---")

        print("  - 阶段 1: 批量进行检索决策...")
        decision_prompts = [create_decision_prompt(r["problem_data"]["problem"], r["generated_steps"][-1] if r["generated_steps"] else None) for r in active_requests]
        decision_outputs = llm.generate(decision_prompts, sampling_params_decision)

        content_prompts = []
        for i, request in enumerate(tqdm(active_requests, desc="  - 处理决策并准备执行")):
            decision_raw = decision_outputs[i].outputs[0].text.strip()
            needs_retrieval = "<retrieval>" in decision_raw
            retrieved_context = None
            retrieval_query = None
            if needs_retrieval:
                problem_text = request["problem_data"]["problem"]
                query_context = request['generated_steps'][-1] if request['generated_steps'] else ""
                query = f"{problem_text}\n{query_context}" if query_context else problem_text
                retrieval_query = query
                retrieved_context = run_retrieval(query, retriever, index, kb_docs)
            request["inference_trace"].append({
                "step_number": f"{step+1}.1_decision", "generator_prompt": decision_prompts[i],
                "raw_generator_output": decision_raw, "needs_retrieval_for_next_step": needs_retrieval,
                "retrieval_query": retrieval_query, "retrieved_context_for_next_step": retrieved_context
            })
            content_prompts.append(create_content_prompt(request["problem_data"]["problem"], request["generated_steps"][-1] if request["generated_steps"] else None, retrieved_context))

        print("  - 阶段 2: 批量生成步骤内容...")
        content_outputs = llm.generate(content_prompts, sampling_params_content)

        for i, request in enumerate(tqdm(active_requests, desc="  - 处理生成内容")):
            output_obj = content_outputs[i].outputs[0]
            raw_content = output_obj.text.strip()
            is_end_of_solution = "[END_OF_SOLUTION]" in raw_content
            content_to_parse = raw_content.split("[END_OF_SOLUTION]")[0].strip()
            parsed_step_obj = parse_json_robust(content_to_parse, request['problem_id'])
            request["generated_steps"].append(parsed_step_obj)
            is_parsing_error = "error" in parsed_step_obj
            request["inference_trace"][-1].update({
                "execution_prompt": content_prompts[i], "execution_output_raw": raw_content,
                "parsed_step_content": parsed_step_obj, "is_end_of_solution": is_end_of_solution,
                "finish_reason": output_obj.finish_reason
            })
            if is_parsing_error or is_end_of_solution or len(request["generated_steps"]) >= MAX_STEPS:
                request["finished"] = True

    print("\n所有推理步骤完成或达到最大步数。")
    with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f_out:
        for request in tqdm(request_states, desc="写入日志文件"):
            final_generated_answer = extract_final_answer_robust(request["generated_steps"]) if request["generated_steps"] else None
            log_entry = {
                "problem_id": request["problem_id"], 
                "problem_text": request["problem_data"]["problem"],
                "gold_answer": request["problem_data"]["answer"], 
                "final_generated_answer": final_generated_answer,
                "generated_steps": request["generated_steps"], 
                "inference_trace": request["inference_trace"],
            }
            f_out.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    print(f"\n推理完成。结果已记录到: {OUTPUT_LOG_PATH}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" not in str(e): raise e
        print("Start method already set.")
    run_inference_vllm()