import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

# --- 1. 定义参数 ---

@dataclass
class ModelArguments:
    """与模型相关的参数"""
    model_name_or_path: str = field(
        default="",
        metadata={"help": ""}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": ""}
    )

@dataclass
class DataArguments:
    """与数据相关的参数"""
    train_file: str = field(
        default="",
        metadata={"help": ""}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": ""}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """自定义训练参数"""
    output_dir: str = field(
        default="",
        metadata={"help": ""}
    )
    num_train_epochs: float = field(default=5.0, metadata={"help": ""})
    per_device_train_batch_size: int = field(default=8, metadata={"help": ""})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": ""})
    learning_rate: float = field(default=1e-5, metadata={"help": ""})
    warmup_ratio: float = field(default=0.1, metadata={"help": ""})
    logging_dir: str = field(default='/data/yangcheng/aaai/trainlogs/retriever_logs', metadata={"help": ""})
    logging_steps: int = field(default=100, metadata={"help": ""})
    save_steps: int = field(default=10000, metadata={"help": ""})
    fp16: bool = field(default=False, metadata={"help": ""})
    bf16: bool = field(
        default=False,
        metadata={"help": " "}
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": ""}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": ""}
    )


# --- 2. 自定义数据整理器 (Data Collator) ---

class ContrastiveDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        texts = []
        group_sizes = []  # <--- 新增: 用于记录每个样本的元素数量
        for feature in features:
            # 计算当前样本的总元素数 (1个query + N个pos + M个neg)
            num_elements = 1 + len(feature['pos']) + len(feature['neg'])
            group_sizes.append(num_elements)

            texts.append(feature['query'])
            texts.extend(feature['pos'])
            texts.extend(feature['neg'])

        batch = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 将group_sizes列表也放入batch中，以便Trainer可以访问
        batch['group_sizes'] = group_sizes
        return batch

# --- 3. 自定义训练器 (Trainer) ---

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_sizes = inputs.pop("group_sizes")

        outputs = model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        per_sample_embeddings = torch.split(embeddings, group_sizes, dim=0)

        temp = self.model.config.temperature
        losses = []
        
        # 遍历批次中的每一个样本来独立计算损失
        for sample_embeddings in per_sample_embeddings:
            query_embedding = sample_embeddings[0:1]
            positive_embedding = sample_embeddings[1:2]
            negative_embeddings = sample_embeddings[2:]

            if negative_embeddings.shape[0] == 0:
                continue
            
            # =================== START: 修正逻辑 ===================
            # 将所有计算逻辑移入循环内部
            
            # 组合正例和负例的 embedding 用于计算分数
            # 正例 shape: (1, D), 负例 shape: (num_neg, D)
            # all_embeddings shape: (1 + num_neg, D)
            all_embeddings = torch.cat([positive_embedding, negative_embeddings], dim=0)

            # 计算 query 和所有 candidate (pos+negs) 的相似度分数
            # query_embedding shape: (1, D), all_embeddings shape: (1+num_neg, D)
            # scores shape: (1, 1 + num_neg)
            scores = torch.matmul(query_embedding, all_embeddings.t())
            scores /= temp
            
            # 正例永远在第一个位置 (index 0)
            labels = torch.zeros(1, dtype=torch.long, device=model.device)
            
            # 计算交叉熵损失
            loss = F.cross_entropy(scores, labels)
            losses.append(loss)
            # ==================== END: 修正逻辑 ====================
        
        # 将批次中所有样本的损失求平均值
        if not losses:
            # 如果整个批次都没有有效的损失（例如，都没有负例）
            # 返回一个需要梯度的零张量，以避免在某些情况下（如只有一个样本且无负例）崩溃
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        final_loss = torch.stack(losses).mean()
        
        return (final_loss, {"outputs": outputs}) if return_outputs else final_loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if torch.cuda.is_available():
        # 获取当前进程被分配到的本地GPU索引
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 检查该GPU的能力
        major, _ = torch.cuda.get_device_capability(local_rank)
        if major >= 8:
            print(f"进程 {local_rank}: 检测到支持BF16的GPU (Compute Capability {major}.x)。将 bf16 设为 True。")
            training_args.bf16 = True
        else:
            print(f"进程 {local_rank}: GPU (Compute Capability {major}.x) 不支持BF16。将 bf16 设为 False。")
            training_args.bf16 = False
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.logging_dir:
        os.makedirs(training_args.logging_dir, exist_ok=True)
    
    print(f"从 {data_args.train_file} 加载数据...")
    train_dataset = load_dataset('json', data_files={'train': data_args.train_file})['train']
    print(f"数据集加载完成，共 {len(train_dataset)} 条样本。")

    print(f"从本地路径加载模型: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # 根据训练参数确定加载模型的数据类型
    torch_dtype = (
        torch.bfloat16 if training_args.bf16 else (
            torch.float16 if training_args.fp16 else torch.float32
        )
    )
    
    # --- 关键改动: 移除Flash Attention ---
    # BGE模型 (基于BertModel) 不支持Flash Attention 2，所以我们不再尝试启用它。
    # 我们仍然会从 BF16/FP16 的使用中获益。
    print(f"模型加载配置: torch_dtype={torch_dtype}, torch_compile={training_args.torch_compile}")

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype
        # 注意: 已移除 attn_implementation 参数，让 transformers 库为 BertModel 使用其默认的注意力机制
    )
    model.config.temperature = model_args.temperature
    
    if torch.cuda.is_available():
        print(f"CUDA可见设备: {os.getenv('CUDA_VISIBLE_DEVICES')}. 将使用GPU。")
    else:
        print("未发现CUDA设备，将在CPU上运行。")
    
    data_collator = ContrastiveDataCollator(tokenizer, data_args.max_seq_length)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("开始训练...")
    trainer.train()

    print(f"训练完成。正在将最终模型保存到 {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("模型保存完毕。")

if __name__ == "__main__":
    main()