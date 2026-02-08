import os
import argparse

# 尽早解析GPU参数
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gpu_id", type=str, default="0", help="The ID of the GPU to use.")
args_early, remaining_argv = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args_early.gpu_id
print(f"--- Early setup: CUDA_VISIBLE_DEVICES set to '{args_early.gpu_id}' ---")

import json
import re
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from vllm import LLM, SamplingParams

MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# --- 【关键修正】添加缺失的函数 ---
def parse_steps_from_knowledge_base(entry):
    """从知识库条目中解析步骤"""
    if not isinstance(entry, dict):
        return []
    steps = []
    step_keys = sorted(
        [k for k in entry.keys() if k.startswith("Step")],
        key=lambda k: int(re.search(r'Step (\d+)', k).group(1))
    )
    for key in step_keys:
        steps.append(entry[key])
    return steps

def extract_mmlu_index_from_text(text: str) -> str:
    if not text: return None
    match = re.search(r'\[Fully supported\].*?(?:The final answer is|The answer is).*?(\d)\b', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'(?:The final answer is|The answer is|####)\s*(\d)\b', text, re.IGNORECASE)
    if match: return match.group(1).strip()
    return None

def extract_open_domain_answer(text: str) -> str:
    if not text: return None
    match_gsm = re.search(r'####\s*([0-9,.\s]+)', text)
    if match_gsm: return match_gsm.group(1).strip().replace(",", "")
    match_box = re.search(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text, re.DOTALL)
    if match_box: return match_box.group(1).strip()
    match_final = re.search(r'(?:The final answer is|The answer is)\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match_final: return match_final.group(1).strip()
    lines = text.strip().splitlines()
    if lines: return lines[-1].strip()
    return text.strip()

def extract_final_answer(generated_text: str, dataset_type: str) -> str:
    if 'mmlu' in dataset_type.lower():
        return extract_mmlu_index_from_text(generated_text)
    else:
        return extract_open_domain_answer(generated_text)
        
def get_gold_answer(item, dataset_type):
    ds_type = dataset_type.lower()
    if 'mmlu' in ds_type:
        answer = item.get('answer')
        return str(answer) if answer is not None else ""
    elif 'gsm8k' in ds_type:
        raw_answer = item.get('answer', "")
        match = re.search(r'####\s*([0-9,.]+)', raw_answer)
        return match.group(1).strip().replace(",", "") if match else raw_answer
    else: # aime, math500, theoremqa
        return str(item.get('answer', '') or item.get('Answer', '') or item.get('solution', ''))

class Retriever:
    def __init__(self, model_path, corpus_path, device):
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)
        self.corpus = load_jsonl(corpus_path)
        self._build_index()
    def _build_index(self):
        print("Building FAISS index...")
        corpus_texts = [entry.get("Problem", "") for entry in self.corpus]
        embeddings = self.model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True, device=self.device)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print("FAISS index built.")
    def retrieve(self, query_text, k):
        query_embedding = self.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        query_embedding_np = query_embedding if isinstance(query_embedding, np.ndarray) else query_embedding.cpu().numpy()
        if query_embedding_np.ndim == 1: query_embedding_np = query_embedding_np[None, :]
        _, indices = self.index.search(query_embedding_np.astype('float32'), k)
        return [self.corpus[i] for i in indices[0]]

def create_self_rag_prompt(problem_text, model_type, dataset_type, choices=None):
    if 'mmlu' in dataset_type.lower():
        options_str = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        problem_full_text = f"{problem_text}\n\nChoices:\n{options_str}"
        answer_format_instruction = "Generate a step-by-step thinking process that leads to the correct answer choice. Conclude with 'The final answer is #### <index>' where <index> is the 0-based index."
    else:
        problem_full_text = problem_text
        if 'gsm8k' in dataset_type.lower():
             answer_format_instruction = "Generate a step-by-step thinking process that leads to the final answer. Conclude with 'The final answer is #### <number>'."
        else:
             answer_format_instruction = "Generate a step-by-step thinking process that leads to the final answer. Conclude with the final answer enclosed in \\boxed{}."
    user_content = f"{problem_full_text}\n\n{answer_format_instruction}"
    if model_type == 'llama2':
        return f"<s>[INST] {user_content} [/INST]"
    elif model_type == 'llama3':
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == 'qwen':
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    else: raise ValueError(f"Unsupported model_type: {model_type}")

def get_special_token_ids(tokenizer):
    tokens = ["[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in tokens}
    if any(v is None or v == tokenizer.unk_token_id for v in token_ids.values()):
        print("Warning: Some special reflection tokens are not in the tokenizer's vocabulary.")
        print({k:v for k,v in token_ids.items() if v is None or v == tokenizer.unk_token_id})
    return token_ids

def run_inference(args):
    print(f"Main inference function starting. Torch device: {DEVICE}")
    os.makedirs(os.path.dirname(args.output_log_path), exist_ok=True)

    print("Loading SELF-RAG generator model...")
    llm = LLM(model=args.generator_model_path, tensor_parallel_size=1, max_model_len=4096, trust_remote_code=True,
              gpu_memory_utilization=0.7, dtype="bfloat16")
    tokenizer = llm.get_tokenizer()
    special_token_ids = get_special_token_ids(tokenizer)
    
    retriever = Retriever(args.bge_model_path, args.retrieval_corpus_path, DEVICE)
    
    test_problems = load_jsonl(args.test_set_path)
    final_results = []
    
    for i, p in enumerate(tqdm(test_problems, desc=f"Running SELF-RAG on {args.dataset_type}")):
        problem_text = p.get("question") or p.get("problem") or p.get("Question") or ""
        choices = p.get("choices")
        gold_answer = get_gold_answer(p, args.dataset_type)
        
        initial_prompt = create_self_rag_prompt(problem_text, args.model_type, args.dataset_type, choices)
        
        sampling_params_decision = SamplingParams(max_tokens=5, temperature=0.0, logprobs=5)
        decision_output = llm.generate([initial_prompt], sampling_params_decision)[0]
        
        do_retrieve = False
        if decision_output.outputs[0].logprobs:
            top_logprobs_dict = decision_output.outputs[0].logprobs[0]
            logprob_retrieval = top_logprobs_dict.get(special_token_ids["[Retrieval]"])
            logprob_no_retrieval = top_logprobs_dict.get(special_token_ids["[No Retrieval]"])
            prob_retrieval = np.exp(logprob_retrieval.logprob) if logprob_retrieval is not None else 0.0
            prob_no_retrieval = np.exp(logprob_no_retrieval.logprob) if logprob_no_retrieval is not None else 0.0
            do_retrieve = prob_retrieval > prob_no_retrieval
        
        generated_text = ""
        
        if do_retrieve:
            evidences = retriever.retrieve(problem_text, k=args.num_retrieved)
            prompts_for_generation = [
                f"{initial_prompt}[Retrieval]<paragraph>{ev.get('Problem', '')}\n{parse_steps_from_knowledge_base(ev)[0] if parse_steps_from_knowledge_base(ev) else ''}</paragraph>" for ev in evidences
            ]
            sampling_params_generate = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, logprobs=5)
            parallel_outputs = llm.generate(prompts_for_generation, sampling_params_generate)
            
            best_output_text, best_score = "", -1
            for output in parallel_outputs:
                if not output.outputs[0].logprobs: continue
                logprobs_dict = output.outputs[0].logprobs[0]
                
                logprob_relevant = logprobs_dict.get(special_token_ids["[Relevant]"])
                prob_relevant = np.exp(logprob_relevant.logprob) if logprob_relevant is not None else 0.0
                
                if prob_relevant > 0.5:
                    logprob_supported = logprobs_dict.get(special_token_ids["[Fully supported]"])
                    logprob_partial = logprobs_dict.get(special_token_ids["[Partially supported]"])
                    prob_supported = np.exp(logprob_supported.logprob) if logprob_supported is not None else 0.0
                    prob_partial = np.exp(logprob_partial.logprob) if logprob_partial is not None else 0.0
                    score = prob_supported + 0.5 * prob_partial
                    if score > best_score:
                        best_score, best_output_text = score, output.outputs[0].text
            generated_text = best_output_text
        else:
            prompt_no_retrieval = f"{initial_prompt}[No Retrieval]"
            sampling_params_generate = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
            generated_text = llm.generate([prompt_no_retrieval], sampling_params_generate)[0].outputs[0].text
        
        final_generated_answer = extract_final_answer(generated_text, args.dataset_type)
        log_entry = {"problem_id": i, "problem_text": problem_text, "gold_answer": gold_answer, "final_generated_answer": final_generated_answer,
                     "full_generated_output": generated_text}
        if 'mmlu' in args.dataset_type.lower(): log_entry["choices"] = choices
        final_results.append(log_entry)

    with open(args.output_log_path, 'w', encoding='utf-8') as f_out:
        for entry in final_results:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"\nInference complete. Log file saved to: {args.output_log_path}")

if __name__ == "__main__":
    full_parser = argparse.ArgumentParser(description="Run SELF-RAG Baseline experiments for short-form tasks.")
    full_parser.add_argument("--generator_model_path", type=str, required=True)
    full_parser.add_argument("--model_type", type=str, required=True, choices=['llama2', 'llama3', 'qwen'])
    full_parser.add_argument("--dataset_type", type=str, required=True, choices=['aime', 'gsm8k', 'math500', 'mmlu-college', 'mmlu-highschool', 'theoremqa'])
    full_parser.add_argument("--bge_model_path", type=str, default="/data/yangcheng/bge-large-en-v1.5")
    full_parser.add_argument("--retrieval_corpus_path", type=str, default="/data/yangcheng/aaai/data/traindata/structuredata/sampled_testbase_structured.jsonl")
    full_parser.add_argument("--test_set_path", type=str, required=True)
    full_parser.add_argument("--output_log_path", type=str, required=True)
    full_parser.add_argument("--num_retrieved", type=int, default=5)
    full_parser.add_argument("--faiss_index_path", type=str, required=False, help="Path to a pre-built FAISS index. If not provided, one will be built in memory.")
    full_parser.add_argument("--gpu_id", type=str, default=args_early.gpu_id)
    args = full_parser.parse_args(remaining_argv)
    run_inference(args)