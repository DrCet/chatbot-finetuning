
from dataclasses import dataclass, field
import sys
import os
import logging
from transformers.trainer_utils import is_main_process
import datasets
from datasets import DatasetDict, load_dataset, ClassLabel, Sequence

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoModelForImageClassification,
    AutoImageProcessor,
    Trainer
)
import transformers
logger = logging.getLogger(__name__)



@dataclass
class ImageClassModelArguments:
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
class ImageClassTrainingArguments(TrainingArguments):
    pass
@dataclass
class DataTrainingArguments:
    dataset_name_or_path: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_subset_name: str = field(
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
    image_column_name: str = field(
        default=None,
        metadata={
            "help": "The column name of the image in the dataset."
        },
    )
    label_column_name: str = field(
        default=None,
        metadata={
            "help": "The column name of the text in the dataset."
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
class DataCollatorWithPadding:
    '''
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        processor ([`PreTrainedProcessor`]): The processor used for processing the inputs.

    '''
    def __init__(self, image_processor,  data_args, label2id):
        self.image_processor = image_processor
        self.data_args = data_args
        self.label2id = label2id
    def __call__(self, batch):
        out = {}
        
        images = batch[self.data_args.image_column_name]
        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt",
        ).pixel_values

        labels = batch[self.data_args.label_column_name]
        labels = [self.label2id[label] for label in labels]
        out['pixel_values'] = pixel_values
        out["labels"] = labels
        return batch
    
def main():
    # 1. Parse the arguments
    # We now keep distinct sets of args for a clean separation of concerns.
    parser = HfArgumentParser((ImageClassModelArguments, DataTrainingArguments, ImageClassTrainingArguments))
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
            data_args.dataset_name_or_path,
            data_args.dataset_subset_name,
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
    if data_args.label_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.label_column_name} not found in dataset.")

    label_feature = raw_datasets["train"].features[data_args.label_column_name]

    # Handle different label feature types
    if isinstance(label_feature, ClassLabel):
        # Predefined labels (e.g., from Hugging Face datasets)
        label_list = label_feature.names
    elif isinstance(label_feature, Sequence):
        # Handle nested sequence of labels
        label_feature = label_feature.feature
        if isinstance(label_feature, ClassLabel):
            label_list = label_feature.names
        else:
            # Collect unique labels from all splits
            label_set = set()
            for split in raw_datasets.values():
                for example in split:
                    label_set.add(example[data_args.label_column_name])
            label_list = sorted(label_set)
    else:
        # Collect unique labels from all splits (generic case)
        label_set = set()
        for split in raw_datasets.values():
            for example in split:
                label_set.add(example[data_args.label_column_name])
        label_list = sorted(label_set)

    # Create mappings
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}


    # 4. Load processor and config
    # The config is here for the future use 

    config = AutoConfig.from_pretrained(
        model_args.config_name_or_path if model_args.config_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(label2id)
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name_or_path,
        cache_dir=model_args.cache_dir,

    )

    # 5. Preprocess the datasets
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    # 7. Load pretrained model
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        cache_dir=model_args.cache_dir,
    )

    # 8. Save config and processor
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            image_processor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 9. Define data collator
    data_collator = DataCollatorWithPadding(
        image_processor=image_processor,
        data_args=data_args,
        label2id=label2id
    ) 

    # 10. Training and evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["validation"] if training_args.do_eval else None,
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