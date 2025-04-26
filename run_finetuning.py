import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import  Dict, List, Optional, Union
import shutil

import datasets
import torch
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed, is_wandb_available
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm


import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
)

from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint, is_main_process

if is_wandb_available():
    import wandb

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        }
    )
    lr_decay: float = field(
        default=0.999875,
        metadata={"help": "Learning rate decay, used with `ExponentialLR` when `do_step_schedule_per_epoch`."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."}
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

def log_text_predictions(trackers, inputs, generated_texts, target_texts, epoch):
    for tracker in trackers:
        if tracker.name == "tensorboard":
            for i in range(min(len(inputs), 10)):
                tracker.writer.add_text(
                    f"sample_{i}", f"Input: {inputs[i]}\nTarget: {target_texts[i]}\nOutput: {generated_texts[i]}", epoch
                )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "text_predictions": [
                        wandb.Html(
                            f"<b>Input:</b> {inputs[i]}<br><b>Target:</b> {target_texts[i]}<br><b>Output:</b> {generated_texts[i]}"
                        )
                        for i in range(min(len(inputs), 10))
                    ]
                }
            )
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

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # 4. Load dataset
    raw_datasets = DatasetDict()

    # Load training dataset
    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load evaluation dataset
    if training_args.do_eval:
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
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
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        verbose=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: {}".format(tokenizer.pad_token))

    # 6. Preprocess the datasets
    forward_attention_mask = True
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
            revision=model_args.model_revision,  # Fixed typo
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
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
        is_seq2seq=is_seq2seq,
        forward_attention_mask=forward_attention_mask,
    )

    # 10. Set up accelerate
    project_name = data_args.project_name
    train_dataset = vectorized_datasets["train"]
    eval_dataset = vectorized_datasets.get("validation", None)

    # inspired from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    # and https://github.com/huggingface/community-events/blob/main/huggan/pytorch/cyclegan/train.py

    logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    per_device_train_batch_size = (
        training_args.per_device_train_batch_size if training_args.per_device_train_batch_size else 1
    )
    total_batch_size = (
        per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 11. Define train_dataloader and eval_dataloader
    train_dataloader = None
    if training_args.do_train:
        sampler = (
            LengthGroupedSampler(
                batch_size=per_device_train_batch_size,
                dataset=train_dataset,
                lengths=train_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=not training_args.group_by_length,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=sampler,
        )

    eval_dataloader = None
    if training_args.do_eval:
        eval_sampler = (
            LengthGroupedSampler(
                batch_size=training_args.per_device_eval_batch_size,
                dataset=eval_dataset,
                lengths=eval_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=eval_sampler,
        )

    # Scheduler and training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps == -1:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        training_args.learning_rate,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
        eps=training_args.adam_epsilon,
    )

    num_training_steps = training_args.max_steps * accelerator.num_processes
    num_warmup_steps = training_args.get_warmup_steps(num_training_steps)
    if training_args.do_step_schedule_per_epoch:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
        )
    else:
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps if training_args.warmup_steps > 0 else 0,
            num_training_steps=num_training_steps,
        )

    # Prepare with accelerator
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    )

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = training_args.to_sanitized_dict()
        accelerator.init_trackers(project_name, tracker_config)

    # Log training setup
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    global_step = 0
    first_epoch = 0

    # Handle checkpoint resumption
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            training_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, training_args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, int(training_args.num_train_epochs)):
        if training_args.do_step_schedule_per_epoch:
            lr_scheduler.step()

        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if not training_args.do_step_schedule_per_epoch:
                    lr_scheduler.step()
                loss_gathered = accelerator.gather(loss.repeat(per_device_train_batch_size)).mean()
                train_loss += loss_gathered.item() / training_args.gradient_accumulation_steps
                accelerator.log({"train_loss": train_loss, "epoch": epoch}, step=global_step)
                train_loss = 0.0
                global_step += 1
                progress_bar.update(1)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step % training_args.eval_steps == 0 and training_args.do_eval and eval_dataloader:
                logger.info("Running validation...")
                model.eval()
                total_eval_loss = 0
                num_batches = 0
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            return_dict=True,
                        )
                        val_loss = outputs.loss
                        total_eval_loss += accelerator.reduce(val_loss).item()
                        num_batches += 1

                    if model_args.model_type == "seq2seq" and training_args.predict_with_generate:
                        generated_ids = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_length=training_args.generation_max_length,
                            num_beams=training_args.generation_num_beams,
                        )
                        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        target_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                        inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                        if accelerator.is_main_process:
                            log_text_predictions(accelerator.trackers, inputs, generated_texts, target_texts, epoch)

                avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
                accelerator.log({"eval_loss": avg_eval_loss, "epoch": epoch}, step=global_step)
                model.train()

            if training_args.save_steps > 0 and global_step % training_args.save_steps == 0:
                if accelerator.is_main_process:
                    if training_args.save_total_limit is not None:
                        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= training_args.save_total_limit:
                            for removing_checkpoint in checkpoints[:len(checkpoints) - training_args.save_total_limit + 1]:
                                shutil.rmtree(os.path.join(training_args.output_dir, removing_checkpoint))
                    save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            if global_step >= training_args.max_steps:
                break

        if global_step >= training_args.max_steps:
            break

    # Final evaluation
    if training_args.do_eval and eval_dataloader:
        logger.info("Running final validation...")
        model.eval()
        total_eval_loss = 0
        num_batches = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                val_loss = outputs.loss
                total_eval_loss += accelerator.reduce(val_loss).item()
                num_batches += 1

            if model_args.model_type == "seq2seq" and training_args.predict_with_generate:
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=training_args.generation_max_length,
                    num_beams=training_args.generation_num_beams,
                )
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                if accelerator.is_main_process:
                    log_text_predictions(accelerator.trackers, inputs, generated_texts, target_texts, epoch)

        avg_eval_loss = total_eval_loss / num_batches if num_batches > 0 else 0.0
        accelerator.log({"final_eval_loss": avg_eval_loss}, step=global_step)

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub:
            model.push_to_hub(training_args.hub_model_id)

    accelerator.end_training()
    if training_args.push_to_hub:
        tokenizer.push_to_hub(training_args.hub_model_id)

    logger.info("***** Training / Inference Done *****")

if __name__ == "__main__":
    main()