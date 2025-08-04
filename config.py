MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 512

ALPACA_DATASET = "tatsu-lab/alpaca"
RLHF_DATASET  = "Dahoas/full-hh-rlhf"

SFT_CONFIG = {
    "output_dir": "./results/sft",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "learning_rate": 2e-4,
}

DPO_CONFIG = {
    "output_dir": "./results/dpo",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "learning_rate": 5e-7,
    "beta": 0.1,
}

PPO_CONFIG = {
    "output_dir": "./results/ppo",
    "learning_rate": 1.41e-5,
    "batch_size": 8,
    "steps": 200,
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}