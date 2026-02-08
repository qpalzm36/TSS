
corpus_file= # jsonl
save_dir=
retriever_name= # this is for indexing naming
retriever_model=

echo "Starting index building process for BGE model..."
echo "Source Corpus: $corpus_file"
echo "Model Path: $retriever_model"
echo "Saving Index to: $save_dir"
# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --faiss_type Flat \
    --save_embedding

echo "Index building complete!"
