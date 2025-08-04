import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from config import MODEL_NAME, SFT_CONFIG, LORA_CONFIG, MAX_LENGTH
from data_utils import load_alpaca_dataset, prepare_tokenizer, print_dataset_sample

def main():
    gc.collect()
    torch.cuda.empty_cache()

    print("=== Starting Supervised Fine-Tuning (SFT) ===")
    print("SFT teaches the model to follow instructions using supervised learning")
    
    print("\n1. Loading tokenizer...")
    tokenizer = prepare_tokenizer(MODEL_NAME)
    
    print("\n2. Loading Alpaca dataset...")
    train_dataset = load_alpaca_dataset(tokenizer, MAX_LENGTH)
    print_dataset_sample(train_dataset, "Alpaca")
    
    print("\n3. Setting up quantization config...")
    """bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )"""
    
    print("\n4. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        #quantization_config=bnb_config,
    )

    print("\n5. Setting up LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\n6. Setting up training...")
    sft_config = SFTConfig(**SFT_CONFIG)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
    )
    
    print("\n7. Starting SFT training...")
    print("This will teach the model to follow instructions!")
    trainer.train()
    
    print("\n8. Saving SFT model...")
    trainer.save_model()
    tokenizer.save_pretrained(SFT_CONFIG["output_dir"])
    
    print("\n=== SFT Training Complete! ===")
    print(f"Model saved to: {SFT_CONFIG['output_dir']}")
    print("The model now knows how to follow basic instructions.")
    
if __name__ == "__main__":
    main()