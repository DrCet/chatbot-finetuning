# Chatbot Finetuning

A pipeline for easy fine-tuning of chat models using Hugging Face models and datasets.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Initialize the pipeline
chatbot-finetuning init

# Select model and dataset
chatbot-finetuning select --model meta-llama/Llama-3-8b --dataset open-assistant/conversations

# Fine-tune the model
chatbot-finetuning train --output-dir ./finetuned_model
```

## Project Structure

- `chatbot_finetuning/`: Core package with CLI and fine-tuning logic.
- `config/`: Configuration files.
- `core/`: Modules for data preprocessing, model loading, and training.
- `tests/`: Unit tests.
- `scripts/`: Utility scripts.
- `docs/`: Documentation.

## License

MIT