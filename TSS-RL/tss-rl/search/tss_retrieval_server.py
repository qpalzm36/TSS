import json
import os
import argparse
import faiss
import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoConfig, AutoTokenizer, AutoModel
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class BGEEncoder:
    def __init__(self, model_path, device="cuda"):
        print(f"Loading BGE model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, queries: List[str]) -> np.ndarray:

        if not queries:
            return np.array([], dtype=np.float32)

        processed_queries = [f"Represent this sentence for searching relevant passages: {q}" for q in queries]
        
        encoded_input = self.tokenizer(processed_queries, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        model_output = self.model(**encoded_input)

        sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()


class TSSRetriever:
    def __init__(self, index_path, corpus_path, model_path, device="cuda:0"):
        self.device = device
        
 
        print(f"Loading FAISS index to CPU from: {index_path}")
        self.index = faiss.read_index(index_path) 

        
        self.encoder = BGEEncoder(model_path, device)
        

        self.corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.corpus.append(json.loads(line))
        print(f"Loaded {len(self.corpus)} docs. Index is running on CPU.")
    def search(self, queries: List[str], topk: int = 1):
        if not queries:
            print("Received empty queries list, skipping...")
            return [] 

        truncated_queries = [q[-2048:] for q in queries]
    

        embeddings = self.encoder.encode(truncated_queries)
        

        scores, indices = self.index.search(embeddings, topk)
        

        results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(topk):
                idx = indices[i][j]
                score = scores[i][j]
                if idx < 0 or idx >= len(self.corpus):
                    continue
                
                doc = self.corpus[idx]
                

                content_text = doc.get('contents', json.dumps(doc, ensure_ascii=False))
                
                query_results.append({
                    'title': f"Example_{idx}", 
                    'text': content_text,      
                    'score': float(score)
                })
            results.append(query_results)
        return results


app = FastAPI()
retriever = None 

class QueryRequest(BaseModel):
    queries: List[str]
    topk: int = 1

@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    results = retriever.search(request.queries, request.topk)

    return {"result": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to BGE model")
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to Knowledge Base JSONL")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for retrieval model (e.g., cuda:0)")
    
    args = parser.parse_args()
    
    retriever = TSSRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.model_path,
        device=args.device
    )
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)