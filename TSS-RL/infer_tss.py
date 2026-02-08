import argparse
import json
import re
import requests
import time
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import List, Dict

# --- 1. 配置与提示词 (与训练时保持绝对一致) ---
SYSTEM_PROMPT = (
    "## Role\n"
    "You are an expert math solver using a 'decision-search-reasoning' loop. Precision is key.\n\n"
    
    "## TSS Process Protocol\n"
    "1. Decide: At each step, output <no-retrieval> or <retrieval>.\n"
    "2. Search: If <retrieval>, output <search>Problem + Last Step Content</search> and STOP. Wait for the system to provide an example in <p>...</p> tags.\n"
    "3. Reason: After receiving the <p> example, or if you chose <no-retrieval>, output your reasoning as a SINGLE JSON object.\n\n"
    
    "## Example: How to transition from Step 1 to Step 2\n"
    "User: Find y if x=2 and y=x+5.\n\n"
    "Response 1 (Step 1):\n"
    "<no-retrieval> {\"Step 1\": \"[CONDITION] x=2. [PROCESS] Substitute x into y=x+5. [CONCLUSION] y=2+5.\"}\n\n"
    "Response 2 (Step 2 Search):\n"
    "<retrieval> <search>Find y if x=2 and y=x+5. [CONDITION] x=2. [PROCESS] Substitute x into y=x+5. [CONCLUSION] y=2+5.</search>\n\n"
    "System Feedback:\n"
    "<p>{\"Problem\": \"Find z if a=3, z=a+2\", \"Step 2\": \"[CONDITION] a=3. [PROCESS] 3+2=5. [CONCLUSION] z=5.\"}</p>\n\n"
    "Response 2 (Step 2 Reasoning):\n"
    "{\"Step 2\": \"[CONDITION] Need to compute 2+5. [PROCESS] 2+5=7. [CONCLUSION] y=7. \\boxed{7} [END_OF_SOLUTION]\"}\n\n"
    
    "## Rules\n"
    "1. The <search> query MUST be the ACTUAL text of the Problem + your Last Step. No placeholders.\n"
    "2. JSON keys must be \"Step 1\", \"Step 2\", etc. Stop immediately after '}'.\n"
    "3. Include \\boxed{answer} and [END_OF_SOLUTION] in the final step."
)

# --- 2. 检索辅助函数 ---
def call_retriever(query: str, url: str, topk: int = 1) -> str:
    """调用检索服务器获取例题"""
    if not query or len(query) < 5:
        return "No query provided."
    
    payload = {"queries": [query], "topk": topk}
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("result", [])
        if result and result[0]:
            # 获取第一个结果的内容 (text字段)
            return result[0][0].get("text", "").strip()
    except Exception as e:
        print(f"[Retrieval Error]: {e}")
    
    return "No example found."

# --- 3. 核心推理类 ---
class TSSInference:
    def __init__(self, model_path, retrieval_url):
        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1, # 根据你的显卡数量调整，如果是单卡推理设为1
            trust_remote_code=True,
            gpu_memory_utilization=0.7, # 推理时可以给多一点
            max_model_len=4096, # 保持与训练一致
            enforce_eager=True  # 避免 CUDAGraph 问题，对于变长生成更稳定
        )
        
        self.retrieval_url = retrieval_url
        
        # 采样参数：Greedy Search 用于评估最稳
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=500, # 单步最大长度
            stop=['}', '</search>', '[END_OF_SOLUTION]'] # 关键停止词
        )
        self.tokenizer = self.llm.get_tokenizer()

    def solve_one_problem(self, problem_text, max_turns=12):
        """
        对单个问题执行 TSS 闭环推理
        """
        # 构造初始对话上下文
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        
        # 将消息转换为 Prompt String (因为我们要手动拼接后续步骤)
        # 注意：这里假设使用 ChatML 格式，或者直接拼接文本
        # 为了精确控制，我们手动拼接 System + User
        current_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        trace = []
        full_log = ""
        
        for step_idx in range(max_turns):
            # 1. 模型生成 (直到遇到停止词)
            outputs = self.llm.generate(
                [current_prompt], 
                self.sampling_params, 
                use_tqdm=False
            )
            generated_text = outputs[0].outputs[0].text
            finish_reason = outputs[0].outputs[0].finish_reason
            
            # 记录这一步的原始输出
            step_record = {"step": step_idx + 1, "raw_output": generated_text, "action": "unknown"}
            
            # --- 2. 解析与分支逻辑 ---
            
            # Case A: 检索 (模型输出了 <search>...)
            if '<retrieval>' in generated_text or '<search>' in generated_text:
                step_record["action"] = "search"
                
                # 提取 Query
                query_match = re.search(r'<search>(.*?)(?:</search>|$)', generated_text, re.DOTALL)
                query = query_match.group(1).strip() if query_match else ""
                
                # 补全生成的标签 (因为可能被 stop 截断了)
                if '</search>' not in generated_text:
                    generated_text += '</search>'
                
                # 调用检索
                retrieved_content = call_retriever(query, self.retrieval_url)
                step_record["query"] = query
                step_record["retrieved"] = retrieved_content
                
                # 拼接回 Prompt (模拟环境反馈)
                # 格式：[生成的内容] + \n\n<p>[例题]</p>\n\n
                current_prompt += f"{generated_text}\n\n<p>{retrieved_content}</p>\n\n"
                
            # Case B: 结束 (模型输出了 [END_OF_SOLUTION])
            elif '[END_OF_SOLUTION]' in generated_text:
                step_record["action"] = "finish"
                current_prompt += generated_text
                trace.append(step_record)
                break # 结束循环
                
            # Case C: 推理 (模型输出了 JSON)
            elif '{' in generated_text:
                step_record["action"] = "reason"
                
                # 确保 JSON 闭合 (因为可能被 '}' 截断)
                if '}' not in generated_text:
                    generated_text += '}'
                
                # 拼接到 Context，准备下一步
                current_prompt += f"{generated_text}\n"
                
            # Case D: 其他 (乱码或空)
            else:
                step_record["action"] = "error"
                # 强行换行，尝试救回来
                current_prompt += f"{generated_text}\n"

            trace.append(step_record)
            
            # 安全检查：防止无限循环
            if len(trace) >= max_turns:
                break

        return {
            "problem": problem_text,
            "final_prompt": current_prompt,
            "trace": trace
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    # data_path 变成可选的
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data (json, jsonl, parquet)")
    # 新增：直接输入问题
    parser.add_argument("--input_query", type=str, default=None, help="Directly input a math problem string to solve")
    parser.add_argument("--output_path", type=str, default="inference_output.jsonl", help="Path to save results")
    parser.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8001/retrieve")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # 初始化推理引擎
    engine = TSSInference(args.model_path, args.retrieval_url)
    
    data = []

    # --- 场景 A: 命令行直接测试单条 ---
    if args.input_query:
        print(f"Running single inference on input: {args.input_query}")
        data.append({"problem": args.input_query, "id": "manual_test"})
    
    # --- 场景 B: 从文件读取 ---
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        
        # 1. JSONL 格式 (一行一个 JSON)
        if args.data_path.endswith('.jsonl'):
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # 兼容不同的 key 名
                        prob = item.get('problem') or item.get('content') or item.get('question')
                        if prob:
                            data.append({"problem": prob, "id": item.get('id', len(data))})

        # 2. JSON 格式 (一个大列表 [{}, {}])
        elif args.data_path.endswith('.json'):
            with open(args.data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for item in items:
                    prob = item.get('problem') or item.get('content') or item.get('question')
                    if prob:
                        data.append({"problem": prob, "id": item.get('id', len(data))})
        
        # 3. Parquet 格式 (保留兼容性)
        elif args.data_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(args.data_path)
            for _, row in df.iterrows():
                try:
                    # 尝试从 verl 的复杂 prompt 结构中提取，或者直接读 problem 列
                    if 'problem' in row:
                        problem = row['problem']
                    else:
                        # 这是一个兜底逻辑，针对 verl 的 prompt 结构
                        prompts = row['prompt']
                        if isinstance(prompts, np.ndarray): prompts = prompts.tolist()
                        # 假设 prompt 列表里第二个是 user 的问题
                        problem = prompts[1]['content'] 
                        if problem.startswith("Problem: "): problem = problem[9:]
                    
                    data.append({"problem": problem, "id": row.get("id", "unknown")})
                except:
                    continue
    else:
        print("Error: Please provide either --data_path or --input_query")
        exit(1)

    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Loaded {len(data)} samples to process.")

    # --- 执行推理 ---
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            try:
                output = engine.solve_one_problem(item['problem'])
                result_entry = {
                    "id": item['id'],
                    "problem": item['problem'],
                    "trace": output['trace'],
                    "full_text": output['final_prompt']
                }
                
                # 实时写入
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                f_out.flush()
                
                # 如果是命令行单条测试，直接打印出来看看
                if args.input_query:
                    print("\n" + "="*50)
                    print(f"FINAL OUTPUT:\n{output['final_prompt']}")
                    print("="*50 + "\n")
                    
            except Exception as e:
                print(f"Error processing {item['id']}: {e}")

    print(f"Inference complete. Results saved to {args.output_path}")