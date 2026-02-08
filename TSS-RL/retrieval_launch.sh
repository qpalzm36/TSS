
file_path=/data/yangcheng/Search-R1/example
index_file=$file_path/index/bge_Flat.index
corpus_file=$file_path/corpus.jsonl
retriever_name=bge
retriever_path=/data/yangcheng/bge-large-en-v1.5

CUDA_VISIBLE_DEVICES=3 python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
