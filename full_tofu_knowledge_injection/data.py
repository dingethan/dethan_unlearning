"""TOFU QA dataset with official-style answer-only loss masking.

Follows locuslab/tofu data_module.convert_raw_data_to_model_format:
  - prompt = question_start + question + question_end
  - answer = answer_tag + answer
  - labels: question tokens -> -100; answer (+ eos when padded) supervised
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset


def load_tofu_split(tofu_root: str | Path, split: str) -> Dataset:
    path = Path(tofu_root) / split
    if not path.exists():
        raise FileNotFoundError(f"TOFU split not found: {path}")
    obj = load_from_disk(str(path))
    if isinstance(obj, DatasetDict):
        if "train" not in obj:
            raise KeyError(f"Expected 'train' in {path}, got {list(obj.keys())}")
        return obj["train"]
    return obj


def build_prompt_answer(
    question: str,
    answer: str,
    question_start_tag: str,
    question_end_tag: str,
    answer_tag: str,
) -> Tuple[str, str, str]:
    prompt = f"{question_start_tag}{question}{question_end_tag}"
    answer_text = f"{answer_tag}{answer}"
    full_text = prompt + answer_text
    return prompt, answer_text, full_text


def convert_qa_to_tensors(
    tokenizer,
    question: str,
    answer: str,
    max_length: int,
    question_start_tag: str,
    question_end_tag: str,
    answer_tag: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Return input_ids, labels, attention_mask, and debug meta."""
    prompt, answer_text, full_text = build_prompt_answer(
        question, answer, question_start_tag, question_end_tag, answer_tag
    )

    # Official TOFU: count question tokens WITH special tokens (usually BOS).
    num_question_tokens = len(tokenizer.tokenize(prompt, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids: List[int] = list(encoded["input_ids"])
    attention_mask: List[int] = list(encoded["attention_mask"])

    pad_length = max_length - len(input_ids)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id must be set before training")

    # Match locuslab/tofu data_module.convert_raw_data_to_model_format padding.
    if pad_length > 0:
        pad_input_ids = input_ids + [tokenizer.eos_token_id] * pad_length
        pad_attention_mask = attention_mask + [0] * pad_length
        labels = input_ids + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
    else:
        pad_input_ids = input_ids
        pad_attention_mask = attention_mask
        labels = list(input_ids)

    q_mask = min(num_question_tokens, len(labels))
    for i in range(q_mask):
        labels[i] = -100

    meta = {
        "num_question_tokens": num_question_tokens,
        "seq_len_unpadded": len(input_ids),
        "n_supervised": int(sum(1 for x in labels if x != -100)),
        "n_masked": int(sum(1 for x in labels if x == -100)),
        "truncated": len(input_ids) >= max_length and pad_length == 0,
        "prompt_preview": prompt[:120],
        "answer_preview": answer_text[:120],
    }
    return (
        torch.tensor(pad_input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(pad_attention_mask, dtype=torch.long),
        meta,
    )


class TofuQADataset(TorchDataset):
    def __init__(
        self,
        data: Dataset,
        tokenizer,
        max_length: int = 512,
        question_start_tag: str = "Question: ",
        question_end_tag: str = "\n",
        answer_tag: str = "Answer: ",
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_start_tag = question_start_tag
        self.question_end_tag = question_end_tag
        self.answer_tag = answer_tag
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError(f"Expected str question/answer at idx={idx}")
        input_ids, labels, attention_mask, _ = convert_qa_to_tensors(
            self.tokenizer,
            question,
            answer,
            self.max_length,
            self.question_start_tag,
            self.question_end_tag,
            self.answer_tag,
        )
        return input_ids, labels, attention_mask


def custom_data_collator(samples: Sequence[Tuple[torch.Tensor, ...]]):
    input_ids = torch.stack([s[0] for s in samples])
    labels = torch.stack([s[1] for s in samples])
    attention_mask = torch.stack([s[2] for s in samples])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def mask_stats(dataset: TofuQADataset, n: int = 64) -> Dict[str, float]:
    n = min(n, len(dataset))
    supervised, masked, truncated = [], [], 0
    for i in range(n):
        row = dataset.data[i]
        _, labels, _, meta = convert_qa_to_tensors(
            dataset.tokenizer,
            row[dataset.question_key],
            row[dataset.answer_key],
            dataset.max_length,
            dataset.question_start_tag,
            dataset.question_end_tag,
            dataset.answer_tag,
        )
        supervised.append(meta["n_supervised"])
        masked.append(meta["n_masked"])
        truncated += int(meta["truncated"])
    return {
        "samples_checked": n,
        "avg_supervised_tokens": float(sum(supervised) / n),
        "avg_masked_tokens": float(sum(masked) / n),
        "truncated_count": truncated,
        "min_supervised": float(min(supervised)),
        "max_supervised": float(max(supervised)),
    }
