import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import csv

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

def main(questions):
    results = []

    for question in questions:
        test_prompt = f"{question}\n\nResponse:"
        print(f"=== Question: {question} ===\n")

        # Original model
        print("1. ORIGINAL MODEL:")
        model, tokenizer = load_model(MODEL_NAME)
        response_orig = generate(model, tokenizer, test_prompt)
        print(response_orig)
        del model, tokenizer
        torch.cuda.empty_cache()

        print("\n" + "="*50 + "\n")

        # SFT model
        print("2. SFT MODEL:")
        if os.path.exists("./results/sft"):
            model, tokenizer = load_model(MODEL_NAME, "./results/sft")
            response_sft = generate(model, tokenizer, test_prompt)
            print(response_sft)
            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            response_sft = "SFT model not found!"
            print(response_sft)

        print("\n" + "="*50 + "\n")

        # DPO model
        print("3. DPO MODEL:")
        if os.path.exists("./results/dpo"):
            model, tokenizer = load_model(MODEL_NAME, "./results/dpo")
            response_dpo = generate(model, tokenizer, test_prompt)
            print(response_dpo)
            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            response_dpo = "DPO model not found!"
            print(response_dpo)

        print("\n" + "="*50 + "\n")

        results.append({
            "question": question,
            "original_response": response_orig,
            "sft_response": response_sft,
            "dpo_response": response_dpo,
        })

    # Save results to CSV
    with open("responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "original_response", "sft_response", "dpo_response"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    questions = [
        "Explain what is RNN in AI in simple terms.",
        "What is the difference between supervised and unsupervised learning?",
        "How does attention mechanism work in Transformers?",
    ]
    main(questions)
