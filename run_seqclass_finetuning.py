import logging
from dataclasses import dataclass, field
from typing import List, Dict, Union
import torch
import torch.nn as nn
import sys
import os

logger = logging.getLogger(__name__)

import datasets
from datasets import load_dataset, DatasetDict
import transformers
from transformers.trainer_utils import is_main_process
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer
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
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to cache directory for storing downloaded models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer or not"}
    )
    overwrite_vocab: bool = field(
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
    preprocessing_num_workers: int = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "If set, only preprocess the dataset and exit."}
    )
    



class SeqClassificationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features: Dict[str, Union[List[str], torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract the text and labels from the features
        texts = [feature['input_ids'] for feature in features]
        labels = torch.tensor([feature['label'] for feature in features])

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
    parser = HfArgumentParser((SeqClassificationModelArguments, SeqClassificationDataArguments, SeqClassificationTrainingArguments))
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

    # 3. Load the dataset
    raw_dataset = DatasetDict() 
    if training_args.do_train:
        raw_dataset["train"] = load_dataset(
            data_args.dataset_name_or_path, 
            data_args.dataset_subset_name, 
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir
        )
    if training_args.do_eval and data_args.eval_split_name is not None:
        raw_dataset['validation'] = load_dataset(
            data_args.dataset_name_or_path, 
            data_args.dataset_subset_name, 
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir
        )
    elif not data_args.eval_split_name:
        new_dataset = raw_dataset["train"].train_test_split(test_size=0.15)
        raw_dataset["train"] = new_dataset["train"]
        raw_dataset["validation"] = new_dataset["test"]

    if data_args.text_column_name not in next(iter(raw_dataset.values())).column_names:
        raise ValueError(f"Text column {data_args.text_column_name} not found in dataset. Available columns: {next(iter(raw_dataset.values()))[0].keys()}")
    if data_args.label_column_name not in next(iter(raw_dataset.values())).column_names:
        raise ValueError(f"Label column {data_args.label_column_name} not found in dataset. Available columns: {next(iter(raw_dataset.values()))[0].keys()}")
    
    # 4. Load the tokenizer, config 
    # The config is used to set the number of labels for the classification task
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # 5. Preporcess the dataset
    with training_args.main_process_first():
        if data_args.max_train_samples is not None:
            raw_dataset["train"] = raw_dataset["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None:
            raw_dataset["validation"] = raw_dataset["validation"].select(range(data_args.max_eval_samples))

    unique_labels = set(raw_dataset["train"][data_args.label_column_name])
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    config.num_labels = len(unique_labels)
    config.label2id = label2id
    config.id2label = id2label

    def prepare_dataset(batch):
        texts = batch[data_args.text_column_name]
        labels = batch[data_args.label_column_name]
        labels = [label2id[label] for label in labels]

        tokenized_batch = tokenizer(
            texts,
            return_tensors=None
        )

        return {
            'input_ids': tokenized_batch['input_ids'],
            'attention_mask': tokenized_batch['attention_mask'],
            'labels': labels
        }
    
    remove_columns = raw_dataset["train"].column_names
    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_dataset.map(
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

    # 6. Load the model
    class SeqClassificationModel(nn.Model): 
        def __init__(self, config=config, dropout_prob=0.1):
            super().__init__()
            self.pretrained_model = AutoModel.from_pretrained(
                model_args.model_name_or_path if model_args.model_name_or_path else model_args.config_name,
                config=config,
                cache_dir=model_args.cache_dir,
            )
            if model_args.overwrite_vocab:
                self.pretrained_model.resize_token_embeddings(len(tokenizer))

            self.dropout = nn.Dropout(dropout_prob)
            self.hidden_layer = nn.Linear(self.config.hidden_size, config.num_labels)
            self.softmax = nn.Softmax(dim=-1)
            self._init_weights(self.hidden_layer)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            # Get outputs from the pretrained model
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

            pooled_output = self.dropout(outputs[1])
            hidden_output = self.relu(self.hidden_layer(pooled_output))
            logits = self.classifier(hidden_output)
            probs = self.softmax(logits)  
            
            loss = None
            if labels is not None:
                loss_fct = nn.NLLLoss()  # NLLLoss expects log-probabilities
                # Convert probabilities to log-probabilities
                log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
                loss = loss_fct(log_probs.view(-1, self.config.num_labels), labels.view(-1))
            
            # Return logits for metrics compatibility, loss if training
            return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
    model = SeqClassificationModel(config=config)
    # 7. Save config
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            config.save_pretrained(training_args.output_dir)

    # 8. Create the data collator
    data_collator = SeqClassificationCollator(tokenizer)
    # 9 . Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # 10. Train the model
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()

    

