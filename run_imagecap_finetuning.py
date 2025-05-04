
from dataclasses import dataclass, field
import sys
import os
from typing import Any, List, Dict, Union
from huggingface_hub import resume_inference_endpoint
import torch
import logging
from transformers.trainer_utils import is_main_process
import datasets
from datasets import DatasetDict, load_dataset

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    Trainer
)
import transformers
logger = logging.getLogger(__name__)



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
    overwrite_vocabulary: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the vocabulary."}
    )

@dataclass
class ImageCapTrainingArguments(TrainingArguments):
    pass
@dataclass
class DataTrainingArguments:
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
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only preprocess the dataset and not train the model."
        },
    )
    train_split_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training split in the dataset."
        },
    )
    eval_split_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation split in the dataset."
        },
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for preprocessing."}
    )
@dataclass
class DataCollatorWithPadding:
    '''
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        processor ([`PreTrainedProcessor`]): The processor used for processing the inputs.

    '''
    def __init__(self, image_processor, tokenizer, forward_attention_mask: bool = True):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.forward_attention_mask = forward_attention_mask
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        for i, f in enumerate(features):
            if 'input_ids' not in f:
                logger.error(f"Missing input_ids in feature {i}: {f.keys()}")
        pixel_values = torch.stack([torch.tensor(f['pixel_values']) for f in features])
        input_ids = [f['input_ids'] for f in features]
        batch = self.tokenizer.pad(
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
            logger.warning(f"Unknown architecture {architecture}.")
            return None

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

    # 3. Load dataset
    raw_datasets = DatasetDict()
    if training_args.do_train:
        raw_datasets['train'] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir
        )
    if training_args.do_eval and data_args.eval_split_name is not None:
        raw_datasets['validation'] = load_dataset(
            data_args.dataset_name_or_path, 
            data_args.dataset_subset_name, 
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir
        )
    elif not data_args.eval_split_name:
        new_dataset = raw_datasets["train"].train_test_split(test_size=0.15)
        raw_datasets["train"] = new_dataset["train"]
        raw_datasets["validation"] = new_dataset["test"]

    if data_args.image_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.image_column_name} not found in dataset.")
    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.text_column_name} not found in dataset.")

    # 4. Load processor and config
    # The config is here for the future use 
    config = AutoConfig.from_pretrained(
        model_args.config_name_or_path if model_args.config_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name_or_path,
        cache_dir=model_args.cache_dir,

    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )


    # 5. Preprocess the datasets
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        images = batch[data_args.image_column_name]
        texts = [str(t) if t is not None else "" for t in batch[data_args.text_column_name]]
        image_inputs = image_processor(images=images, return_tensors=None)
        text_inputs = tokenizer(texts, return_tensors=None)
        logger.info(f"Image inputs: {image_inputs}")
        logger.info(f"Text inputs: {text_inputs}")
        return {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"] ,
            "attention_mask": text_inputs["attention_mask"]
        }

    remove_columns = raw_datasets["train"].column_names
    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            batched=True,
            num_proc=data_args.preprocessing_num_workers or os.cpu_count(),
            remove_columns=remove_columns,
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
            model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            logger.warning(f"Failed to resize token embeddings: {e}.")
            raise e
    # 8. Save config and processor
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            image_processor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 9. Define data collator
    data_collator = DataCollatorWithPadding(
        image_processor=image_processor, tokenizer=tokenizer
    ) 

    # 10. Training and evaluation
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