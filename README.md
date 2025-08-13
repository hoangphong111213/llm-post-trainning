# llm-post-trainning
# LLM Fine-tuning Pipeline

A simple pipeline to fine-tune language models using SFT and DPO training.

## What it does

Fine-tunes **Qwen/Qwen3-0.6B** model through:

1. **SFT Training** - Teaches the model to follow instructions (Alpaca dataset)
2. **DPO Training** - Aligns the model with human preferences (HH-RLHF dataset)
3. **Evaluation** - Compares original vs SFT vs DPO models

## Files

- `config.py` - All settings and hyperparameters
- `data_utils.py` - Data loading functions
- `1_sft_training.py` - Run SFT training
- `2_dpo_training.py` - Run DPO training  
- `3_compare_models.py` - Compare all models

## Training Data

- **SFT**: [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) - 52K instruction-following examples
- **DPO**: [HH-RLHF Dataset](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) - Human preference pairs (limited to 10K samples)

## Sample Results

The pipeline shows clear improvement across training stages:

| Stage | Quality | Instruction Following | Coherence |
|-------|---------|----------------------|-----------|
| Original | Basic | Poor | Low |
| SFT | Good | Strong | High |
| DPO | Excellent | Excellent | Very High |

Example comparison for "Explain what is RNN in AI in simple terms":

- **Original**: Fragmented explanation with unrelated Q&A format
- **SFT**: Clear, structured explanation covering key concepts
- **DPO**: Comprehensive explanation with practical applications and context

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python 1_sft_training.py    # First run SFT
python 2_dpo_training.py    # Then run DPO
python 3_compare_models.py  # Compare results
```

## Results

The pipeline improves model quality:
- **Original**: Basic, unfocused responses
- **SFT**: Better instruction following
- **DPO**: Human-aligned, high-quality responses

### Sample Results (responses.csv)

| Question | Original Response | SFT Response | DPO Response |
|----------|-------------------|--------------|--------------|
| Explain what is RNN in AI in simple terms | "The RNN is a type of neural network used to process and understand sequential data... [fragmented with unrelated Q&A]" | "RNN is a type of deep learning algorithm that uses a sequence of inputs to produce an output... [clear explanation]" | "RNN stands for recurrent neural networks and is a type of artificial neural network used to learn patterns in time-based data... [comprehensive]" |
| What is the difference between supervised and unsupervised learning? | "The difference between supervised and unsupervised learning... [includes confusing multiple choice format]" | "Supervised learning is a type of machine learning where the model learns to recognize patterns... [structured explanation]" | "Supervised learning is a type of machine learning that uses labeled data... [clear, focused explanation]" |

Complete results are saved in `responses.csv`.


## Configuration

Edit `config.py` to change:
- Model name (default: Qwen/Qwen3-0.6B)
- Training epochs
- Batch sizes
- Learning rates
- LoRA parameters
