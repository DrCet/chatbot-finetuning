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
from monotonic_align import maximum_path
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
from transformers.trainer_pt_utils import LenghGroupedSampler
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import send_example_telemetry


if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)