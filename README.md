# Chatbot Finetuning

A pipeline for fine-tuning chat models like `distilgpt2` on datasets like `wikitext` or `cnn_dailymail` using Hugging Face Transformers and Accelerate. This project includes scripts to initialize models (`create_model.py`) and fine-tune them (`run_finetuning.py`) with customizable configurations.

## Features
- Initialize models from the Hugging Face Hub and save locally or push to a new repository.
- Fine-tune causal language models (e.g., `distilgpt2`) or seq2seq models with distributed training.
- Support for logging training metrics (TensorBoard, Weights & Biases).
- Configurable via a JSON file (`finetune.json`).
- Optimized for Kaggle GPU environments with memory-efficient settings.

## Installation

### Prerequisites
- Python 3.8+ (tested with 3.11 on Kaggle).
- Hugging Face account and API token for Hub integration.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/DrCet/chatbot-finetuning
   cd chatbot-finetuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note*: If running on Kaggle, there is no need to install dependencies

## Usage


### 0. Loggin to HuggingFace
When using notebook environment, run:
```
from huggingface_hub import notebook_login
notebook_login()
```

or
```
from huggingface_hub import login
login(token='YOUR TOKEN')
```

### 1. Initialize a Model
Use `create_model.py` to download a model and tokenizer from the Hugging Face Hub and save them locally or push to a new repository.

```bash
python create_model.py \
  --model_name distilbert/distilgpt2 \
  --pytorch_dump_folder_path ./model_output \
  --model_type causal_lm \
  --repo_id your_username/distilgpt2-finetuned-wikitext
```

- `--model_name`: Model ID (e.g., `distilbert/distilgpt2`).
- `--pytorch_dump_folder_path`: Directory to save model and tokenizer.
- `--model_type`: `causal_lm` (default) or `seq2seq`.
- `--repo_id`: (Optional) Hugging Face Hub repository ID to push the model.

*Note*: Requires `HF_TOKEN` environment variable or Kaggle secret for `--repo_id`.

### 2. Fine-Tune the Model
Use `run_finetuning.py` to fine-tune the model on a dataset, configured via `finetune.json`.

```bash
accelerate launch run_finetuning.py ./finetune.json
```

- `finetune.json`: Configuration file (see below).

### Configuring `finetune.json`
Create a `finetune.json` file in the project root to specify training parameters. Below is an example configuration:

```json
{
    "model_name_or_path": "distilbert/distilgpt2",
    "model_type": "causal-lm",
    "use_fast_tokenizer": true,
    "override_vocabulary_embeddings": false,
    "dataset_name": "roneneldan/TinyStories",
    "text_column_name": "text",
    "max_tokens_length": 128,
    "overwrite_cache": true,
    "processing_num_workers": 4,
    "do_lower_case": false,
    "output_dir": "./test_output",
    "do_train": true,
    "do_eval": true,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 1,
    "max_steps": 20,
    "save_steps": 10,
    "eval_steps": 10,
    "logging_steps": 5,
    "overwrite_output_dir": true,
    "report_to": "tensorboard",
    "seed": 42,
    "gradient_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "learning_rate": 5e-5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "do_step_schedule_per_epoch": false,
    "save_total_limit": 3,
    "fp16": true,
    "cache_dir": "./test_output/cache"
  }
```

