import os
import token
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from datasets import load_from_disk, concatenate_datasets
from dataclasses import dataclass, field
import math
import numpy as np
from llm_unlearn.evaluation.mia import compute_mia
from llm_unlearn.methods import adversarial_training
from llm_unlearn.utils import tokenize, smart_tokenizer_and_embedding_resize

logger = logging.get_logger(__name__)


@dataclass
class UnlearningArguments(TrainingArguments):
    do_unlearn: bool = field(default=False, metadata={
                             "help": "Whether to run unlearning."})
    do_unlearn_eval: bool = field(default=False, metadata={
                                  "help": "Whether to run unlearning eval."})
    unlearn_method: str = field(default="retrain", metadata={
                                "help": "The method used to perform unlearning."})
    completely_random: bool = field(
        default=False, metadata={"help": "Whether to use completely random."}
    )
    top_k: int = field(
        default=1e10, metadata={"help": "top k sample to generate random label"}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "top p sample to generate random label"}
    )
    use_soft_labels: bool = field(
        default=False,
        metadata={"help": "Whether to use soft labels for training."}
    )
    rm_groundtruth: bool=field(
        default=False,
        metadata={"help": "Whether to remove ground truth when sampling random labels."}
    )
    unlearned_model_name_or_path: str =field(
        default = None,
        metadata={"help": "The unlearned model."}
    )
    domain: str =field(
        default = None,
        metadata={"help": "The unlearned domain."}
    )
    
