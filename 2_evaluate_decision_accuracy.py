import os
import json
import re
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# --- 配置 ---
# 使用 argparse 来接收模型路径和模型类型，使其更灵活
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of the retrieval decision model.")
    parser.add_argument("--generator_model_path", type=str, required=True, help="Path to the generator model.")
    parser.add_argument("--model_type", type=str, required=True, choices=['llama2', 'llama3', 'qwen'], help="Type of the model for prompt formatting.")
    parser.add_argument("--parsed_data_path", type=str, default="/data/yangcheng/aaai/specialtokenprediction/parsed_decision_eval_data.jsonl", help="Path to the parsed evaluation data.")
    parser.add_argument("--output_dir", type=str, default="/data/yangcheng/aaai/specialtokenprediction/results", help="Directory to save evaluation results.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use.")
    return parser.parse_args()

# --- Prompt 创建函数 ---

def create_llama2_prompt(problem: str, previous_step: dict = None) -> str:
    system_instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no-retrieval>`."
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {json.dumps(previous_step)}")
    user_content = f"{{{', '.join(input_parts)}}}"
    return f"<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n{user_content} [/INST] "

def create_llama3_prompt(problem: str, previous_step: dict = None) -> str:
    instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no-retrieval>`." # Llama3 用 <no-retrieval>
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {json.dumps(previous_step)}")
    input_content = f"{{{', '.join(input_parts)}}}"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{input_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def create_qwen_prompt(problem: str, previous_step: dict = None) -> str:
    instruction = (
        "You are a planner for a math solving agent. Your task is to decide if you need to retrieve an example for the next step. "
        "Based on the problem and the previous step, your response MUST be one of two tags and nothing else: `<retrieval>` or `<no-retrieval>`."
    )
    problem_str = json.dumps(problem)
    input_parts = [f'"problem": {problem_str}']
    if previous_step:
        input_parts.append(f"\"previous_step\": {json.dumps(previous_step)}")
    input_content = f"{{{', '.join(input_parts)}}}"
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_content}<|im_end|>\n"
        f"<|im_start|>assistant\n")

# --- 主逻辑 ---

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("Loading vLLM generator...")
    llm = LLM(
        model=args.generator_model_path,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        enforce_eager=True
    )

    prompt_fn_map = {
        'llama2': create_llama2_prompt,
        'llama3': create_llama3_prompt,
        'qwen': create_qwen_prompt
    }
    create_decision_prompt = prompt_fn_map[args.model_type]
    
    stop_tokens = ["<retrieval>", "<no-retrieval>"]
    if args.model_type == 'llama3':
        pass
        
    sampling_params_decision = SamplingParams(
        max_tokens=8,
        temperature=0.0,
        stop=stop_tokens,
        include_stop_str_in_output=True,
        ignore_eos=True,
        skip_special_tokens=False
    )

    print(f"Loading parsed data from: {args.parsed_data_path}")
    with open(args.parsed_data_path, 'r', encoding='utf-8') as f:
        eval_data = [json.loads(line) for line in f]

    all_prompts = []
    all_labels = []
    
    print("Preparing prompts for all steps...")
    for problem_data in tqdm(eval_data, desc="Processing problems"):
        # 【关键修复】检查steps列表是否为空
        if not problem_data.get("steps"):
            continue
            
        problem_text = problem_data["problem"]
        all_prompts.append(create_decision_prompt(problem_text, None))
        all_labels.append(problem_data["steps"][0]["decision_label"])

        for i in range(len(problem_data["steps"]) - 1):
            previous_step_content = problem_data["steps"][i]["step_content"]
            next_step_label = problem_data["steps"][i+1]["decision_label"]
            all_prompts.append(create_decision_prompt(problem_text, previous_step_content))
            all_labels.append(next_step_label)

    print(f"Total decision points to evaluate: {len(all_prompts)}")
    
    print("Running batch inference...")
    outputs = llm.generate(all_prompts, sampling_params_decision)

    all_predictions = []
    for output in tqdm(outputs, desc="Parsing predictions"):
        raw_pred = output.outputs[0].text.strip()
        if "<retrieval>" in raw_pred:
            all_predictions.append("<retrieval>")
        elif "<no-retrieval>" in raw_pred or "<no retrieval>" in raw_pred:
            all_predictions.append("<no-retrieval>")
        else:
            all_predictions.append("other")
            
    print("\nCalculating metrics...")
    
    valid_indices = [i for i, pred in enumerate(all_predictions) if pred != "other"]
    y_true_str = [all_labels[i] for i in valid_indices]
    y_pred_str = [all_predictions[i] for i in valid_indices]
    
    labels_map = {"<retrieval>": 1, "<no-retrieval>": 0}
    # 【加固】处理标签集中可能存在的未知标签
    y_true = [labels_map[label] for label in y_true_str if label in labels_map]
    y_pred = [labels_map[y_pred_str[i]] for i, label in enumerate(y_true_str) if label in labels_map]

    # ... (后续的指标计算和保存代码不变) ...
    # 确保 y_true 和 y_pred 长度一致且不为空
    if not y_true:
        print("No valid labels found to calculate metrics.")
        return

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true, y_pred, labels=[1, 0], zero_division=0)

    # 准备结果报告
    num_other = len(all_predictions) - len(valid_indices)
    results = {
        "model_path": args.generator_model_path,
        "total_samples": len(all_prompts),
        "valid_predictions": len(valid_indices),
        "invalid_predictions (other)": num_other,
        "overall_accuracy": accuracy,
        "retrieval_precision": p_class[0],
        "retrieval_recall": r_class[0],
        "retrieval_f1-score": f1_class[0],
        "retrieval_support": int(s_class[0]),
        "no-retrieval_precision": p_class[1],
        "no-retrieval_recall": r_class[1],
        "no-retrieval_f1-score": f1_class[1],
        "no-retrieval_support": int(s_class[1]),
    }
    
    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    
    # 保存结果到文件
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_name = os.path.basename(args.generator_model_path)
    output_file = os.path.join(args.output_dir, f"decision_accuracy_{model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()