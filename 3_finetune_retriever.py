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



@dataclass
class ModelArguments:

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



class ContrastiveDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        texts = []
        group_sizes = []
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
        
        batch['group_sizes'] = group_sizes
        return batch

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_sizes = inputs.pop("group_sizes")

        outputs = model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        per_sample_embeddings = torch.split(embeddings, group_sizes, dim=0)

        temp = self.model.config.temperature
        losses = []
        
        for sample_embeddings in per_sample_embeddings:
            query_embedding = sample_embeddings[0:1]
            positive_embedding = sample_embeddings[1:2]
            negative_embeddings = sample_embeddings[2:]

            if negative_embeddings.shape[0] == 0:
                continue
            
            all_embeddings = torch.cat([positive_embedding, negative_embeddings], dim=0)

            scores = torch.matmul(query_embedding, all_embeddings.t())
            scores /= temp
         
            labels = torch.zeros(1, dtype=torch.long, device=model.device)
 
            loss = F.cross_entropy(scores, labels)
            losses.append(loss)

        

        if not losses:

            return torch.tensor(0.0, device=model.device, requires_grad=True)

        final_loss = torch.stack(losses).mean()
        
        return (final_loss, {"outputs": outputs}) if return_outputs else final_loss

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if torch.cuda.is_available():

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        major, _ = torch.cuda.get_device_capability(local_rank)
        if major >= 8:
            training_args.bf16 = True
        else:

            training_args.bf16 = False
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.logging_dir:
        os.makedirs(training_args.logging_dir, exist_ok=True)
    

    train_dataset = load_dataset('json', data_files={'train': data_args.train_file})['train']


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)


    torch_dtype = (
        torch.bfloat16 if training_args.bf16 else (
            torch.float16 if training_args.fp16 else torch.float32
        )
    )
    

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype

    )
    model.config.temperature = model_args.temperature
    
    data_collator = ContrastiveDataCollator(tokenizer, data_args.max_seq_length)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )


    trainer.train()


    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
