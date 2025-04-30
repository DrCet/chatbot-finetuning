
from dataclasses import dataclass, field
import sys
import os
from typing import Any, List, Dict, Union
import torch
import logging
from transformers.trainer_utils import is_main_process
import datasets
from datasets import DatasetDict, load_dataset

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer
)
import transformers
logger = logging.getLogger(__name__)



@dataclass
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
        default=None,  # Added default to fix error
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use the fast tokenizer (backed by the 🤗 Tokenizers library)."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Revision of the model to use (can be a branch name, tag name or git commit id)."}
    )
    token: str = field(
        default=None,
        metadata={"help": "Huggingface token for private models"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token from the cache or login if not already done."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to allow the use of code from the model repo."}
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
        pixel_values = torch.stack([f['pixel_values'] for f in features])
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

def load_model(checkpoint: str, config):
        """
        Automatically load the appropriate model and processor for a given checkpoint.
        
        Args:
            checkpoint (str): Model checkpoint (e.g., "microsoft/git-base", "Salesforce/blip-image-captioning-base").
            device (str): Device to load the model on (e.g., "cuda", "cpu").
        
        Returns:
            tuple: (model, processor)
        """
        # Mapping of model architectures to AutoModel classes
        model_class_mapping = {
            "GitForCausalLM": AutoModelForCausalLM,
            "BlipForConditionalGeneration": AutoModelForVision2Seq,
            "VisionEncoderDecoderModel": AutoModelForVision2Seq,
            "T5ForConditionalGeneration": AutoModelForSeq2SeqLM,
            "BartForConditionalGeneration": AutoModelForSeq2SeqLM,
            "MBartForConditionalGeneration": AutoModelForSeq2SeqLM
        }

        # Load configuration
        architecture = config.architectures[0] if config.architectures else None

        if not architecture:
            raise ValueError(f"No architecture found in config for {checkpoint}")

        # Find the appropriate AutoModel class
        model_class = None
        for arch_name, auto_class in model_class_mapping.items():
            if arch_name in architecture:
                model_class = auto_class
                break

        if model_class is None:
            logger.warning(f"Unknown architecture {architecture}. Defaulting to AutoModelForCausalLM.")
            model_class = AutoModelForCausalLM

        # Load model
        model = model_class.from_pretrained(checkpoint)
        logger.info(f"Loaded model with architecture {architecture} using {model_class.__name__}")

        return model

def main():
    # 1. Parse the arguments
    # We now keep distinct sets of args for a clean separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ImageCapTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Otherwise we use the command line arguments to parse them.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Set up logging
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

    logger.setLevel(logging.INFO if is_main_process(training_args) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


    #3. Detecting last checkpoint and eventually continue from it
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    # 4. Load dataset
    raw_datasets = DatasetDict()
    if training_args.do_train:
        raw_datasets['train'] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token
        )
    if training_args.do_eval:
        raw_datasets['validation'] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token
        )
    if data_args.image_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.image_column_name} not found in dataset.")
    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.text_column_name} not found in dataset.")

    # 5. Load processor and config
    config = AutoConfig.from_pretrained(
        model_args.config_name_or_path if model_args.config_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )


    # 6. Preprocess the datasets
    forward_attention_mask = True
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        images = batch[data_args.image_column_name]
        texts = batch[data_args.text_column_name]
        inputs = processor(images=images, text=texts, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].split(1),  # List of [1, C, H, W]
            "input_ids": inputs["input_ids"].split(1),       # List of [1, seq_len]
            "attention_mask": inputs["attention_mask"].split(1)
        }

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

    # 7. Load pretrained model
    model = load_model(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name_or_path,
        config=config
    )
    if model_args.overwrite_vocabulary:
        try:
            model.resize_token_embeddings(len(processor.tokenizer))
        except Exception as e:
            logger.warning(f"Failed to resize token embeddings: {e}.")
            raise e
    # 8. Save configs
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            processor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)
    # 9. Define data collator
    data_collator = DataCollatorWithPadding(
        processor=processor,
        forward_attention_mask=forward_attention_mask,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["validation"] if training_args.do_eval else None,
        data_collator=data_collator,
    )
    # 10. Training and evaluation
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint is not None else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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