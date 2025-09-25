import json
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# --- 路径配置 ---
# 1. 输入: 结构化的数据文件
STRUCTURED_DATA_PATH = ""

# 2. 输入: 我们微调好的检索器模型路径
FINETUNED_RETRIEVER_PATH = ""

# 3. 输出: 知识库的保存路径
KB_DIR = ""
# 3.1. 存储知识库所有文档的jsonl文件
KNOWLEDGE_BASE_DOCS_PATH = os.path.join(KB_DIR, "knowledge_base_docs.jsonl")
# 3.2. 存储Faiss索引的文件
FAISS_INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")


def create_knowledge_base_documents(structured_data_path, output_docs_path):
    """
    从结构化数据生成知识库文档。
    每个文档都是一个 "Problem -> Step i -> Step i+1" 的逻辑链条。
    """
    print(f"开始从 {structured_data_path} 生成知识库文档...")
    
    os.makedirs(os.path.dirname(output_docs_path), exist_ok=True)
    
    docs = []
    with open(structured_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            problem = data.get('problem')
            if not problem:
                continue

            # --- 关键修改 ---
            # 从顶层JSON对象中直接提取所有步骤
            steps_items = []
            for key, value in data.items():
                if key.startswith('Step '):
                    steps_items.append((key, value))
            
            # 如果没有找到步骤，则跳过此条目
            if not steps_items:
                continue

            # 将步骤按 "Step 1", "Step 2" ... 排序
            # 使用try-except来处理可能出现的"Final Conclusion"等非标准步骤键
            def get_step_num(item):
                try:
                    return int(item[0].split(' ')[1])
                except (ValueError, IndexError):
                    return float('inf') # 将非标准步骤排到最后

            sorted_steps = sorted(steps_items, key=get_step_num)
            
            # --- 新增: 添加 "问题 -> 步骤1" 的文档 ---
            if sorted_steps:
                step_1_key, step_1_val = sorted_steps[0]
                # 确保第一个确实是 "Step 1"
                if 'Step ' in step_1_key:
                    doc = {
                        "Problem": problem,
                        step_1_key: step_1_val
                    }
                    docs.append(doc)

            # 遍历所有步骤，除了最后一个
            for i in range(len(sorted_steps) - 1):
                step_i_key, step_i_val = sorted_steps[i]
                step_i_plus_1_key, step_i_plus_1_val = sorted_steps[i+1]

                # 确保我们不会创建 "Step X -> Final Conclusion" 这样的无效对
                if 'Step ' not in step_i_plus_1_key:
                    continue
                
                # 构造与检索器正例完全相同的格式
                doc = {
                    "Problem": problem,
                    step_i_key: step_i_val,
                    step_i_plus_1_key: step_i_plus_1_val
                }
                docs.append(doc)

    with open(output_docs_path, 'w', encoding='utf-8') as f_out:
        for doc in docs:
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
    print(f"知识库文档生成完毕，共 {len(docs)} 条文档，已保存至 {output_docs_path}")
    return docs

def build_faiss_index(model_path, docs_path, index_path, max_length=512):
    """
    使用微调好的模型对文档进行编码，并构建FAISS索引。
    """
    if not os.path.exists(docs_path) or os.path.getsize(docs_path) == 0:
        print(f"错误: 文档文件 {docs_path} 不存在或为空。无法构建索引。")
        return

    print(f"开始构建FAISS索引...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    with open(docs_path, 'r', encoding='utf-8') as f:
        documents_text = [line.strip() for line in f]

    print(f"开始对 {len(documents_text)} 条文档进行编码...")
    all_embeddings = []
    with torch.no_grad():
        for text in tqdm(documents_text, desc="Encoding documents"):
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]
            normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            all_embeddings.append(normalized_embedding.cpu().numpy())

    if not all_embeddings:
        print("错误：没有生成任何向量。无法创建FAISS索引。")
        return

    embeddings_np = np.vstack(all_embeddings)
    embedding_dim = embeddings_np.shape[1]
    print(f"编码完成. 向量维度: {embedding_dim}")

    print("构建FAISS索引...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np.astype('float32'))
    
    print(f"索引构建完成，共添加 {index.ntotal} 个向量。")
    faiss.write_index(index, index_path)
    print(f"FAISS索引已成功保存至: {index_path}")


if __name__ == "__main__":
    create_knowledge_base_documents(
        structured_data_path=STRUCTURED_DATA_PATH,
        output_docs_path=KNOWLEDGE_BASE_DOCS_PATH
    )
    
    build_faiss_index(
        model_path=FINETUNED_RETRIEVER_PATH,
        docs_path=KNOWLEDGE_BASE_DOCS_PATH,
        index_path=FAISS_INDEX_PATH
    )
    
    print("\n知识库构建流程全部完成！")
