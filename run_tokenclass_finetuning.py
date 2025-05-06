from transformers.trainer_utils import is_main_process
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
)
from datasets import load_dataset, DatasetDict, ClassLabel, Sequence
import datasets
from dataclasses import dataclass, field
from typing import Dict, List, Union
import torch
import logging
import sys
import os

logger = logging.getLogger(__name__)


@dataclass
class TokenClassModelArguments:
    model_name_or_path: str = field(
        default="bert-base-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name_or_path: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer or not"},
    )
@dataclass
class TokenClassTrainingArguments(TrainingArguments):
    pass

@dataclass 
class DataTrainingArguments:
    dataset_name_or_path: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_subset_name: str = field(
        default=None,
        metadata={"help": "The name of the configuration to use (via the datasets library)."},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."},
    )
    train_split_name: str = field(
        default=None,
        metadata={"help": "The name of the training split."},
    )
    eval_split_name: str = field(
        default=None,
        metadata={"help": "The name of the evaluation split."},
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the column containing the text."},
    )
    label_column_name: str = field(
        default=None,
        metadata={"help": "The name of the column containing the labels."},
    )
    preprocessing_num_workers: int = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The batch size to use for the preprocessing."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "Whether to only preprocess the data and not train."},
    )

class TokenClassDatacollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, List[int]]:
        input_ids =[f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        encodings = self.tokenizer.pad(
            {'input_ids':input_ids},
            padding=True,
            return_tensors="pt",
        )
        padded_labels = []
        for i, label in enumerate(labels):
            label = label + [0] * (len(encodings['input_ids'][i]) - len(label) - 2)  # Adjust for [CLS], [SEP]
            padded_labels.append([-100] + label + [-100])  # Add -100 for [CLS], [SEP]
        padded_labels = torch.tensor(padded_labels)
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': padded_labels,
        }
def main():
    # 1. Parse the arguments
    # We now keep distinct sets of args for a clean separation of concerns.
    parser = HfArgumentParser((TokenClassModelArguments, DataTrainingArguments, TokenClassTrainingArguments))
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

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.text_column_name} not found in dataset.")
    if data_args.label_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(f"Column {data_args.label_column_name} not found in dataset.")

    # 4. Load processor and config
    # The config is here for the future use 
    config = AutoConfig.from_pretrained(
        model_args.config_name_or_path if model_args.config_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Get label information
    label_feature = raw_datasets["train"].features[data_args.label_column_name]
    if isinstance(label_feature, Sequence):
        label_feature = label_feature.feature  # Get inner feature for Sequence

    if isinstance(label_feature, ClassLabel):
        label_list = label_feature.names
    elif hasattr(label_feature, '_str2int'):
        label_list = list(label_feature._str2int.keys())
    else:
        label_set = set()
        for example in raw_datasets["train"]:
            label_set.update(example[data_args.label_column_name])
        label_list = sorted(label_set)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    config.label2id = label2id
    config.id2label = id2label

    # 5. Preprocess the datasets
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        texts = batch[data_args.text_column_name]
        texts = [list( text) for text in texts]  # Convert to list of strings
        raw_labels = batch[data_args.label_column_name]
        
        text_inputs = tokenizer(
            texts,
            return_tensors=None
        )
        labels = []
        for i, label in enumerate(raw_labels):
            word_ids = text_inputs.word_ids(batch_index=i)  
            aligned_labels = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:  
                    aligned_labels.append(-100)
                elif word_idx != prev_word_idx:  
                    label = id2label[label[word_idx]]  
                    aligned_labels.append(label2id[label])
                else: 
                    aligned_labels.append(-100) 
                prev_word_idx = word_idx
            labels.append(aligned_labels)

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
    model = AutoModelForTokenClassification(
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
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # 9. Define data collator
    data_collator = TokenClassDatacollator(
        tokenizer=tokenizer
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