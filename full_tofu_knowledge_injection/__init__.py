"""Full TOFU knowledge-injection utilities.

This package is ONLY for injecting fictitious TOFU knowledge via LoRA SFT.
"""

from .data import (
    TofuQADataset,
    build_prompt_answer,
    custom_data_collator,
    load_tofu_split,
    mask_stats,
)
from .config import load_yaml_config

__all__ = [
    "TofuQADataset",
    "build_prompt_answer",
    "custom_data_collator",
    "load_tofu_split",
    "mask_stats",
    "load_yaml_config",
]
