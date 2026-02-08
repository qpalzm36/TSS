MODEL_PATH=""


INDEX_PATH=""


CORPUS_PATH=""


export CUDA_VISIBLE_DEVICES=6


python tss-rl/search/tss_retrieval_server.py \
    --model_path $MODEL_PATH \
    --index_path $INDEX_PATH \
    --corpus_path $CORPUS_PATH \
    --port 8001 \
    --device cuda:0 