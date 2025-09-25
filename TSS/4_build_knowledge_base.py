import json
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


STRUCTURED_DATA_PATH = ""


FINETUNED_RETRIEVER_PATH = ""


KB_DIR = ""

KNOWLEDGE_BASE_DOCS_PATH = os.path.join(KB_DIR, "knowledge_base_docs.jsonl")

FAISS_INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")


def create_knowledge_base_documents(structured_data_path, output_docs_path):
  

    
    os.makedirs(os.path.dirname(output_docs_path), exist_ok=True)
    
    docs = []
    with open(structured_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            problem = data.get('problem')
            if not problem:
                continue

      
            steps_items = []
            for key, value in data.items():
                if key.startswith('Step '):
                    steps_items.append((key, value))

            if not steps_items:
                continue

       
            def get_step_num(item):
                try:
                    return int(item[0].split(' ')[1])
                except (ValueError, IndexError):
                    return float('inf') 

            sorted_steps = sorted(steps_items, key=get_step_num)
            
      
            if sorted_steps:
                step_1_key, step_1_val = sorted_steps[0]
         
                if 'Step ' in step_1_key:
                    doc = {
                        "Problem": problem,
                        step_1_key: step_1_val
                    }
                    docs.append(doc)


            for i in range(len(sorted_steps) - 1):
                step_i_key, step_i_val = sorted_steps[i]
                step_i_plus_1_key, step_i_plus_1_val = sorted_steps[i+1]

         
                if 'Step ' not in step_i_plus_1_key:
                    continue
                
      
                doc = {
                    "Problem": problem,
                    step_i_key: step_i_val,
                    step_i_plus_1_key: step_i_plus_1_val
                }
                docs.append(doc)

    with open(output_docs_path, 'w', encoding='utf-8') as f_out:
        for doc in docs:
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
            

    return docs

def build_faiss_index(model_path, docs_path, index_path, max_length=512):
    
    if not os.path.exists(docs_path) or os.path.getsize(docs_path) == 0:
    
        return


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    with open(docs_path, 'r', encoding='utf-8') as f:
        documents_text = [line.strip() for line in f]


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
   
        return

    embeddings_np = np.vstack(all_embeddings)
    embedding_dim = embeddings_np.shape[1]



    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np.astype('float32'))
    

    faiss.write_index(index, index_path)



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
    
