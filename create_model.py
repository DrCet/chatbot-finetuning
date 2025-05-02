import argparse
import os
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger("chatbot_finetuning.model")


def create_model(
    model_name: str,
    pytorch_dump_folder_path: str,
    model_type: str = "causal_lm",
    repo_id: str = None
):
    """
    Create and save a model and tokenizer from the Hugging Face Hub.

    Args:
        model_name: Name of the model on the Hugging Face Hub (e.g., "meta-llama/Llama-3-8b").
        pytorch_dump_folder_path: Directory to save the model and tokenizer.
        model_type: Type of model ("causal_lm", "seq2seq", "text_classification). Defaults to "causal_lm".
        repo_id: Optional repository ID to push the model to the Hugging Face Hub.

    Raises:
        ValueError: If model_name, pytorch_dump_folder_path, or model_type is invalid.
        RuntimeError: If downloading or pushing to the Hub fails.
    """
    # Validate inputs
    if not model_name:
        raise ValueError("model_name must be provided")
    if model_type not in ["causal_lm", "seq2seq", "text_classification"]:
        raise ValueError("model_type must be 'causal_lm' or 'seq2seq'")
    if not os.path.isdir(pytorch_dump_folder_path):
        os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # Select model class
    model_mapping = {
        "causal_lm": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
        "text_classification": AutoModelForSequenceClassification,  # Placeholder, adjust as needed
    }

    model_class = model_mapping.get(model_type)

    # Check if model exists locally
    if os.path.exists(os.path.join(pytorch_dump_folder_path, "config.json")):
        logger.info(f"Loading model from {pytorch_dump_folder_path}")
        try:
            model = model_class.from_pretrained(pytorch_dump_folder_path)
            tokenizer = AutoTokenizer.from_pretrained(pytorch_dump_folder_path)
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    else:
        # Download model and tokenizer
        logger.info(f"Downloading model {model_name} from Hugging Face Hub...")
        try:
            model = model_class.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            raise

    # Save model and tokenizer
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    try:
        model.save_pretrained(pytorch_dump_folder_path)
        tokenizer.save_pretrained(pytorch_dump_folder_path)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    # Push to Hugging Face Hub if repo_id is provided
    if repo_id:
        try:
            model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a model and tokenizer from the Hugging Face Hub")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on the Hugging Face Hub")
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--model_type", type=str, default="causal_lm", choices=["causal_lm", "seq2seq", "text_classification"], help="Type of model")
    parser.add_argument("--repo_id", type=str, default=None, help="Repository ID for pushing to the Hugging Face Hub")

    args = parser.parse_args()
    create_model(args.model_name, args.pytorch_dump_folder_path, args.model_type, args.repo_id)