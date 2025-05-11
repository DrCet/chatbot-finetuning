import logging
import os
import sys
from dataclasses import dataclass, field
from typing import  Dict, List, Optional, Union
import datasets
import torch
import os
from datasets import DatasetDict, load_dataset



import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from transformers.trainer_utils import  is_main_process
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    '''  A dataclass automatically generates __init__, __repr__, and other utility methods.
    It's a clean way to define configuration structures.'''
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_type: str = field(
        default="causal-lm",
        metadata={"help": "Model type to use (causal-lm or seq2seq)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizers or not."}
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
                "The configuration (specific version, subset) name of the dataset to use (from datasets.load_dataset)."
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
    train_split_name: str = field(
        default="train",
        metadata={"help": "Name of the training split."},
    )
    eval_split_name: str = field(
        default="validation",
        metadata={"help": "Name of the evaluation split."},
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


def main():

    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LMTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse arguments from JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse from command line arguments
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # 4. Load dataset
    raw_datasets = DatasetDict()

    # Load training dataset
    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir,
        )

    # Load evaluation dataset
    if training_args.do_eval:
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
        )

    # For CausalLM, make sure text_column_name exists
    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # For Seq2SeqLM, make sure source and target columns exist
    if data_args.source_column_name is not None and data_args.source_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--source_column_name {data_args.source_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--source_column_name` to the correct source column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.target_column_name is not None and data_args.target_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--target_column_name {data_args.target_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--target_column_name` to the correct target column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load tokenizer and config
    if model_args.model_name_or_path:
        # Load configuration from pre-trained model or custom path
        # It has not much use now, but it is here for future use
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: {}".format(tokenizer.pad_token))

    # 6. Preprocess the datasets
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        model_inputs = {}
        if model_args.model_type == "causal-lm":
            texts = batch[data_args.text_column_name]
            if data_args.do_lower_case:
                texts = [text.lower() for text in texts]
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=int(data_args.max_tokens_length),
                return_tensors=None,
            )
            model_inputs["input_ids"] = tokenized["input_ids"]
            model_inputs["tokens_input_length"] = [len(ids) for ids in tokenized["input_ids"]]
        else:
            sources = batch[data_args.source_column_name]
            targets = batch[data_args.target_column_name]
            if data_args.do_lower_case:
                sources = [source.lower() for source in sources]
                targets = [target.lower() for target in targets]
            tokenized_inputs = tokenizer(
                sources,
                truncation=True,
                max_length=int(data_args.max_tokens_length),
                return_tensors=None,
            )
            tokenized_labels = tokenizer(
                targets,
                truncation=True,
                max_length=int(data_args.max_tokens_length),
                return_tensors=None,
            )
            model_inputs["input_ids"] = tokenized_inputs["input_ids"]
            model_inputs["labels"] = tokenized_labels["input_ids"]
            model_inputs["tokens_input_length"] = [len(ids) for ids in tokenized_inputs["input_ids"]]
        return model_inputs

    remove_columns = raw_datasets["train"].column_names
    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            batched=True,
            num_proc=data_args.processing_num_workers or os.cpu_count(),
            remove_columns=remove_columns,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preprocess train dataset",
        )
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    #7. Load pretrained model
    if model_args.model_type == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif model_args.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_args.model_type}. Must be 'causal-lm' or 'seq2seq'.")

    # Resize token embeddings if override_vocabulary_embeddings is set
    with training_args.main_process_first(desc="resize token embeddings"):
        if model_args.override_vocabulary_embeddings:
            new_num_tokens = len(tokenizer)
            logger.info(f"Resizing token embeddings to {new_num_tokens} to match tokenizer vocabulary.")
            model.resize_token_embeddings(new_num_tokens)

    # 8. Save configs
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 9. Define data collator
    is_seq2seq = model_args.model_type == "seq2seq"
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        is_seq2seq=is_seq2seq
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
    )
   
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    

if __name__ == "__main__":
    main()