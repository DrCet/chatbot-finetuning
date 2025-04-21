import logging
import os
import sys
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed, is_wandb_available
from datasets import DatasetDict, load_dataset
from wandb import config
# from monotonic_align import maximum_path
from tqdm.auto import tqdm


import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
)

from transformers.feature_extraction_utils import BatchFeature
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import send_example_telemetry


if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    '''  A dataclass automatically generates __init__, __repr__, and other utility methods.
    It's a clean way to define configuration structures.'''
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizers or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Revision of the model to use (can be a branch name, tag name or git commit id)."},
    )
    token:str = field(
        default=None,
        metadata={"help": "Huggingface token for private models."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    override_speaker_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True` and if `speaker_id_column_name` is specified, it will replace current speaker embeddings with a new set of speaker embeddings."
                "If the model from the checkpoint didn't have speaker embeddings, it will initialize speaker embeddings."
            )
        },
    )

    override_vocabulary_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, it will resize the token embeddings based on the vocabulary size of the tokenizer. In other words, use this when you use a different tokenizer than the one that was used during pretraining."
            )
        },
    )


@dataclass 
class LMTrainingArguments(TrainingArguments):

    predict_with_generate: bool = field(
    default=True,
    metadata={"help": "Use generate() method for evaluation (important for Seq2SeqLM)."}
    )

    generation_max_length: Optional[int] = field(
        default=256,
        metadata={"help": "Max length for sequence generation during evaluation."}
    )

    generation_num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "Beam width for generation."}
    )
    do_step_schedule_per_epoch: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to perform scheduler steps per epoch or per steps. If `True`, the scheduler will be `ExponentialLR` parametrized with `lr_decay`."
            )
        },
    )
    lr_decay: float = field(
        default=0.999875,
        metadata={"help": "Learning rate decay, used with `ExponentialLR` when `do_step_schedule_per_epoch`."},
    )

@dataclass
class DataTrainingArguments:
    ''' Arguments pertaining to what data we are going to input our model for training and eval. '''
    project_name: str = field(
        default="text_generation",
        metadata={"help": "Name of the project for wandb."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (HuggingFace datasets format)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use (from datasets.load_dataset)."
                "If not specified, will be used as the dataset name."
            )
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the cached training and evaluation sets"
            )
        },
    )

    processing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The number of processes to use for the preprocessing."
                "If not specified, will be set to the number of CPU cores."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )

    source_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column name for source/input text (used for Seq2SeqLM)."},
    )

    target_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column name for target/response text (used for Seq2SeqLM)."},
    )

    text_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column name for full input text (used for CausalLM)."},
    )

    max_tokens_length: float = field(
        default=512,
        metadata={"help": "Truncate texts longer than this many tokens."},
    )

    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "Run only preprocessing and skip training."},
    )

    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Lowercase the input text."},
    )

    do_normalize: bool = field(
        default=False,
        metadata={"help": "Normalize input waveform if applicable."},
    )

    train_split_name: str = field(
        default="train",
        metadata={"help": "Name of the training split."},
    )

    eval_split_name: str = field(
        default="validation",
        metadata={"help": "Name of the evaluation split."},
    )

    full_generation_sample_text: str = field(
        default="This is a test input.",
        metadata={"help": "Input prompt for full generation sanity check."},
    )


class DataCollatorForLanguageModeling:
    """
    A unified data collator for CausalLM and Seq2SeqLM models.
    """

    def __init__(self, tokenizer, is_seq2seq: bool = False, forward_attention_mask: bool = True):
        self.tokenizer = tokenizer
        self.is_seq2seq = is_seq2seq
        self.forward_attention_mask = forward_attention_mask

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.is_seq2seq:
            # Separate source and target
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]

            # Pad inputs
            model_inputs = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                return_attention_mask=self.forward_attention_mask,
            )

            # Pad labels separately
            labels_batch = self.tokenizer.pad(
                {"input_ids": labels},
                padding=True,
                return_tensors="pt",
                return_attention_mask=False,
            )
            model_inputs["labels"] = labels_batch["input_ids"]
        else:
            # For CausalLM: labels = input_ids
            input_ids = [f["input_ids"] for f in features]
            model_inputs = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                return_attention_mask=self.forward_attention_mask,
            )
            # For CausalLM, labels are the same as input_ids and the model will ignore the padding tokens
            # The model will shift the input_ids to the right and predict the next token internally
            model_inputs["labels"] = model_inputs["input_ids"].clone()

        return model_inputs


print(f'{__file__} passed')