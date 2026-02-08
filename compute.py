import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import argparse
import os

def parse_log_entry(entry):
    """
    解析日志条目，提取模型名称、方法、K-shot等信息。
    根据您的命名约定进行解析。
    """
    source_file = entry['source_log_file']
    dataset_type = entry['dataset_type']
    accuracy = entry['final_accuracy']

    # 1. 解析基础模型和方法
    method = "TSS" # 默认是您的TSS方法
    base_model = ""
    k_shot_str = ""
    k_shot=""
    # 从文件名中提取信息
    filename = os.path.basename(source_file)
    
    # 尝试匹配 SELF-RAG
    selfrag_match = re.search(r'selfrag(\d+b)_(mmlu|aime|theoremqa|gsm8k|math500)_(\d)\.jsonl', filename)
    if selfrag_match:
        method = "SELF-RAG"
        base_model = f"Llama{selfrag_match.group(1).replace('b','')}-{'7B' if selfrag_match.group(1) == '7b' else '13B'}"
        k_shot_str = selfrag_match.group(3)
        if dataset_type == 'mmlu-college' or dataset_type == 'mmlu-highschool':
            # SELF-RAG论文默认是5-shot，但您的日志文件名是 _3, _4
            # 这里我们根据文件名来判断k
            if k_shot_str == '1': k_shot = 1
            elif k_shot_str == '2': k_shot = 2
            elif k_shot_str == '3': k_shot = 3
            elif k_shot_str == '4': k_shot = 4
            else: k_shot = None # Fallback
        else:
            k_shot = None # 其他数据集的SELF-RAG暂时无法从文件名判断k

    # 尝试匹配 SPELL
    spell_match = re.search(r'SPELL/.*/(llama|qwen)(\d+b)_(\d)/results.jsonl', source_file)
    if spell_match:
        method = "SPELL"
        base_model = f"{spell_match.group(1).capitalize()}{spell_match.group(2).replace('b','-')}B"
        k_shot_str = spell_match.group(3) # 这里是num_selected
        if k_shot_str: k_shot = int(k_shot_str)
        else: k_shot = None

    # 尝试匹配 IDS (使用 /collegebase/ 或 /highbase/ )
    ids_match = re.search(r'(llama|qwen)(\d+b)_(\d)\.jsonl', filename)
    if ids_match and ('collegebase' in source_file or 'highbase' in source_file):
        method = "IDS"
        base_model = f"{ids_match.group(1).capitalize()}{ids_match.group(2).replace('b','-')}B"
        k_shot_str = ids_match.group(3) # 这里是num_examples_to_retrieve
        if k_shot_str: k_shot = int(k_shot_str)
        else: k_shot = None
    
    # 尝试匹配 TSS (您的方法)
    tss_match = re.search(r'(qwen|llama)(\d+b)_(\d)\.jsonl', filename)
    if tss_match and method == "TSS": # 只有在没有被其他方法匹配时才认为是TSS
        method = "TSS"
        base_model = f"{tss_match.group(1).capitalize()}{tss_match.group(2).replace('b','-')}B"
        k_shot_str = tss_match.group(3) # 这里是num_examples_to_retrieve
        if k_shot_str: k_shot = int(k_shot_str)
        else: k_shot = None

    # 匹配无检索 (0-shot)
    if 'no_retrieval' in filename:
        method = "No Retrieval"
        base_model_match = re.search(r'(llama|qwen)(\d+b)_no_retrieval', filename)
        if base_model_match:
            base_model = f"{base_model_match.group(1).capitalize()}{base_model_match.group(2).replace('b','-')}B"
        k_shot = 0

    # 如果还没匹配到，尝试通用模型名提取
    if not base_model:
        model_name_match = re.search(r'(llama2_7b|llama3_8b|qwen_7b|qwen3b)', filename)
        if model_name_match:
            if model_name_match.group(1) == 'llama2_7b': base_model = 'Llama2-7B'
            elif model_name_match.group(1) == 'llama3_8b': base_model = 'Llama3-8B'
            elif model_name_match.group(1) == 'qwen_7b': base_model = 'Qwen-7B'
            elif model_name_match.group(1) == 'qwen3b': base_model = 'Qwen-3B'


    # 确保k_shot不为None
    if k_shot is None: k_shot = 1 # 默认值

    return {
        'dataset': dataset_type,
        'base_model': base_model,
        'method': method,
        'k_shot': k_shot,
        'accuracy': accuracy,
        'source_file': source_file # 用于调试
    }

def generate_typory_table(summary_file_path):
    # 加载所有数据
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        all_entries = [json.loads(line) for line in f]

    # 解析和过滤数据
    parsed_data = []
    for entry in all_entries:
        parsed_entry = parse_log_entry(entry)
        if parsed_entry['base_model'] and parsed_entry['method']: # 确保解析成功
            parsed_data.append(parsed_entry)

    df = pd.DataFrame(parsed_data)

    # 数据清理和标准化
    df['base_model'] = df['base_model'].replace({'qwen_7b': 'Qwen-7B', 'qwen3b': 'Qwen-3B', 'llama2_7b': 'Llama2-7B', 'llama3_8b': 'Llama3-8B'})
    df['k_shot_label'] = df['k_shot'].apply(lambda k: f"{k}-shot" if k > 0 else "0-shot")

    # 对数据进行分组，计算均值和标准差
    grouped_results = df.groupby(['dataset', 'base_model', 'method', 'k_shot_label'])['accuracy'].agg(['mean', 'std']).reset_index()

    # 重新命名列以美化输出
    grouped_results.rename(columns={'mean': 'Accuracy Mean', 'std': 'Accuracy Std'}, inplace=True)

    # 转换为百分比格式
    grouped_results['Accuracy Mean'] = grouped_results['Accuracy Mean'].apply(lambda x: f"{x:.2%}")
    grouped_results['Accuracy Std'] = grouped_results['Accuracy Std'].apply(lambda x: f"±{x:.2%}" if pd.notna(x) else "")
    grouped_results['Accuracy'] = grouped_results['Accuracy Mean'] + " " + grouped_results['Accuracy Std']

    # 选择需要的列
    final_df = grouped_results[['dataset', 'base_model', 'method', 'k_shot_label', 'Accuracy']].copy()

    # 按照数据集、模型、方法和k-shot排序
    sort_order = {
        'No Retrieval': 0, 'TSS': 1, 'IDS': 2, 'SPELL': 3, 'SELF-RAG': 4
    }
    final_df['method_sort_key'] = final_df['method'].map(sort_order)
    final_df.sort_values(by=['dataset', 'base_model', 'method_sort_key', 'k_shot_label'], inplace=True)
    final_df.drop(columns=['method_sort_key'], inplace=True)

    # 生成Typora Markdown表格
    markdown_table = final_df.to_markdown(index=False)
    print(markdown_table)
    
    return markdown_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Typora Markdown table from ALL_EXPERIMENTS_SUMMARY.jsonl.")
    parser.add_argument("--summary_file_path", type=str, required=True, help="Path to the ALL_EXPERIMENTS_SUMMARY.jsonl file.")
    args = parser.parse_args()
    
    generate_typory_table(args.summary_file_path)