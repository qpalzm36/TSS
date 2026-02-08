import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import json
import argparse

def build_index():
    parser = argparse.ArgumentParser(description="Build a FAISS index for a given corpus and model.")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Name of the Hugging Face model to use (e.g., BAAI/bge-base-en-v1.5).")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus.jsonl file.")
    parser.add_argument("--output_path", type=str, default="./bge_index.faiss", help="Path to save the output FAISS index file.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding documents (adjust based on GPU memory).")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for the tokenizer.")
    
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    corpus = []
    try:
        with open(args.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus.append(data.get('contents', ''))
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {args.corpus_path}")
        return
        
    print(f"Loaded {len(corpus)} documents from {args.corpus_path}.")

    print("Encoding all documents...")
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), args.batch_size), desc="Encoding"):
            batch = corpus[i:i + args.batch_size]
            
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=args.max_length).to(device)
            outputs = model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0]
            
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(normalized_embeddings.cpu().numpy())

    doc_vectors = np.concatenate(all_embeddings, axis=0).astype('float32')
    print(f"Finished encoding. Vector shape: {doc_vectors.shape}")

    print("Building FAISS index...")
    vector_dim = doc_vectors.shape[1]
    
    quantizer = faiss.IndexFlatL2(vector_dim)
    
    nlist = int(4 * np.sqrt(doc_vectors.shape[0]))
    index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)

    print("Training FAISS index...")
    index.train(doc_vectors)
    print("Adding vectors to index...")
    index.add(doc_vectors)
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    print(f"Saving index to {args.output_path}...")
    faiss.write_index(index, args.output_path)
    print("All done.")

if __name__ == "__main__":
    build_index()