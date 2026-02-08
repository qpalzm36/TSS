file_path=/the/path/you/save/corpus
corpus_file=

bge_index_file=$file_path/bge_IVFFlat.index

python build_bge_index.py --model_name "BAAI/bge-base-en-v1.5" \
                          --corpus_path $corpus_file \
                          --output_path $bge_index_file