import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import PeftConfig, get_peft_model, PeftModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from config import MODEL_NAME, DPO_CONFIG, MAX_LENGTH
from data_utils import load_rlhf_dataset, prepare_tokenizer, print_dataset_sample

def main():
    gc.collect()
    torch.cuda.empty_cache()
    print("=== Starting Direct Preference Optimization (DPO) ===")
    print("DPO teaches the model human preferences without a reward model")
    
    sft_model_path = "./results/sft"
    if not os.path.exists(sft_model_path):
        print(f"ERROR: SFT model not found at {sft_model_path}")
        print("Please run 1_sft_training.py first!")
        return
    
    print("\n1. Loading tokenizer...")
    tokenizer = prepare_tokenizer(MODEL_NAME)
    
    print("\n2. Loading HH-RLHF preference dataset...")
    train_dataset = load_rlhf_dataset(tokenizer, MAX_LENGTH)
    print_dataset_sample(train_dataset, "HH-RLHF Preferences")
    
    print("\n3. Loading SFT model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    
    peft_config = PeftConfig.from_pretrained("./results/sft")
    peft_config.inference_mode = False

    model = get_peft_model(base, peft_config)
    model.load_adapter("./results/sft", adapter_name="default")
    model.train()
    
    print("\n4. Creating reference model...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    ref_model = get_peft_model(ref_base, peft_config)
    ref_model.load_adapter("./results/sft", adapter_name="default")
    ref_model.train()
    
    print("\n5. Setting up DPO training...")

    dpo_config = DPOConfig(**DPO_CONFIG)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
    )

    print("\n6. Starting DPO training...")
    print("This teaches the model to prefer better responses!")
    trainer.train()
    
    print("\n7. Saving DPO model...")
    trainer.save_model()
    tokenizer.save_pretrained(DPO_CONFIG["output_dir"])
    
    print("\n=== DPO Training Complete! ===")
    print(f"Model saved to: {DPO_CONFIG['output_dir']}")

if __name__ == "__main__":
    main()