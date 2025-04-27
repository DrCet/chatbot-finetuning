
from dataclasses import dataclass, field

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
class LayoutTrainingArguments:
    pass