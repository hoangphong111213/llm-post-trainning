import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from config import MODEL_NAME

def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def main(test_prompt):
    print("=== COMPARING MODELS ===\n")
    
    # Original model
    print("1. ORIGINAL MODEL:")
    model, tokenizer = load_model(MODEL_NAME)
    response = generate(model, tokenizer, test_prompt)
    print(response)
    del model, tokenizer
    torch.cuda.empty_cache()
    
    print("\n" + "="*50 + "\n")
    
    # SFT model
    print("2. SFT MODEL:")
    if os.path.exists("./results/sft"):
        model, tokenizer = load_model(MODEL_NAME, "./results/sft")
        response = generate(model, tokenizer, test_prompt)
        print(response)
        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("SFT model not found!")
    
    print("\n" + "="*50 + "\n")
    
    # DPO model
    print("3. DPO MODEL:")
    if os.path.exists("./results/dpo"):
        model, tokenizer = load_model(MODEL_NAME, "./results/dpo")
        response = generate(model, tokenizer, test_prompt)
        print(response)
        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("DPO model not found!")

if __name__ == "__main__":
    question = "Explain what is RNN in AI in simple terms."
    test_prompt = f"{question}\n\nResponse:"
    main(test_prompt)