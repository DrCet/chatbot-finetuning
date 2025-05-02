import logging
from dataclasses import dataclass, field
from typing import List, Dict, Union
import torch
import sys
import os

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset
import transformers
from transformers.trainer_utils import is_main_process
from transformers import (
    TrainingArguments,
    HFArgumentParser,
)

@dataclass
class SeqClassificationModelArguments:
    """
    Arguments pertaining to the model configuration and checkpoint for sequence classification.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cach_dir: str = field(
        default=None,
        metadata={"help": "Path to cache directory for storing downloaded models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer or not"}
    )
    overwrite_vocalb: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the vocabulary of the model"}
    )

@dataclass
class SeqClassificationTrainingArguments(TrainingArguments):
    pass

@dataclass
class SeqClassificationDataArguments:
    dataset_name_or_path: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_subset_name: str = field(
        default=None,
        metadata={"help": "The subset of the dataset to use."}
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the column containing the text data."}
    )
    label_column_name: str = field(
        default=None,
        metadata={"help": "The name of the column containing the labels."}
    )
    train_split_name: str = field(
        default=None,
        metadata={"help": "The name of the training split."}
    )
    eval_split_name: str = field(
        default=None,
        metadata={"help": "The name of the evaluation split."}
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                          "value if set."}
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                          "value if set."}
    )


class SeqClassificationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features: Dict[str, Union[List[str], torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract the text and labels from the features
        texts = [feature['text'] for feature in features]
        labels = [feature['label'] for feature in features]

        # Tokenize the texts
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', return_tensors='pt')

        # Create a batch dictionary
        batch = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

        return batch


def main():
    # 1. Initialize the arguments
    parser = HFArgumentParser((SeqClassificationModelArguments, SeqClassificationDataArguments, SeqClassificationTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):        
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)