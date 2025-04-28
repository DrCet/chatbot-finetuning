
from dataclasses import dataclass, field
from typing import Any, List, Dict, Union
import torch

from transformers import (
    TrainingArguments
)
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name_or_path: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str = field(
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use the fast tokenizer (backed by the 🤗 Tokenizers library)."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "Revision of the model to use (can be a branch name, tag name or git commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={"help": "Huggingface token for private models"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token from the cache or login if not already done. "
            "It can be used to download private models and datasets."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow the use of code from the model repo. "
            "This may be used to load custom models or layers."
        },
    )
    overwrite_vocabulary: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the vocabulary."}
    )

@dataclass
class ImageCapTrainingArguments(TrainingArguments):
    do_step_schedule_per_epoch: bool = field(
        default=True,
        metadata={"help": "Whether to do step schedule per epoch."}
    )
    lr_decay: float = field(
        default=0.999875,
        metadata={"help": "Learning rate decay, used with `ExponentialLR` when `do_step_schedule_per_epoch`."},
    )

@dataclass
class DataTrainingArguments:
    project_name: str = field(
        default='ImageCationingFinetuning',
        metadata={"help": "The name of the project to use for logging."},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    max_train_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: int = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    image_column_name: str = field(
        default=None,
        metadata={
            "help": "The column name of the image in the dataset."
        },
    )
    text_column_name: str = field(
        default=None,
        metadata={
            "help": "The column name of the text in the dataset."
        },
    )
    max_token_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only preprocess the dataset and not train the model."
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training split in the dataset."
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation split in the dataset."
        },
    )
    do_lower_case: bool = field(   
        default=False,
        metadata={
            "help": "Whether to do lower case for the tokenizer."
        },
    )
    do_normailze: bool = field(
        default=False,
        metadata={
            "help": "Whether to do normalization for the tokenizer."
        },
    )
@dataclass
class DataCollatorWithPadding:
    '''
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        processor ([`PreTrainedProcessor`]): The processor used for processing the inputs.

    '''
    def __init__(self, processor, forward_attention_mask: bool = True):
        self.processor = processor
        self.forward_attention_mask = forward_attention_mask
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f'pixel_values' for f in features])
        input_ids = [f['input_ids'] for f in features]
        batch = self.processor.tokenizer.pad(
           {'input_ids':input_ids},
           return_tensors="pt",
           return_attention_mask=self.forward_attention_mask,
           padding=True
       )
        batch['pixel_values'] = pixel_values
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
def main():
    pass

