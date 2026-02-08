# TSS


**Config** </br>
python=3.10 </br>
pip install openai pandas jsonlines scikit-learn numpy </br>
pip install accelerate matplotlib seaborn tqdm pyarrow </br>
pip install torch transformers sentence-transformers </br>
pip install einops transformers_stream_generator </br>
pip install tiktoken faiss-cpu datasets peft </br>
pip install vllm </br>

**retriever**:fine-tune bge-en-v1.5 </br>

**generator**:fine-tune Qwen-2.5-3B-Instruct,Qwen2.5-Math-7B-Instruct,,Meta-Llama3-8B-Instruct,Llama-2-7b-chat-hf </br>

train genetrator using LLaMA-Factory

Inference files are in the fold: lorainfer

evaluate files are files starting with 9_evaluate
