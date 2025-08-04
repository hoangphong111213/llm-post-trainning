from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_alpaca_dataset(tokenizer, max_length=512):
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    def format_instruction(example):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output
        return {"text": full_text}
    
    dataset = dataset.map(format_instruction)
    
    print(f"Loaded {len(dataset)} samples from Alpaca dataset")
    return dataset

def load_rlhf_dataset(tokenizer, max_length=512):
    print("Loading HH-RLHF dataset...")
    dataset = load_dataset("Dahoas/full-hh-rlhf", split="train")
    dataset = dataset.select(range(10000))
    def format_pairs(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    dataset = dataset.map(format_pairs)
    
    print(f"Loaded {len(dataset)} samples from HH-RLHF dataset")
    return dataset

def prepare_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def print_dataset_sample(dataset, dataset_name, num_samples=2):
    print(f"\n=== {dataset_name} Dataset Samples ===")
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        sample = dataset[i]
        for key, value in sample.items():
            print(f"{key}: {str(value)[:200]}...")
        print("-" * 50)