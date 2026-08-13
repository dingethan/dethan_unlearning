#!/usr/bin/env python3
"""Official-style TOFU evaluation for θ_full and retain reference."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection import load_tofu_split, load_yaml_config
from full_tofu_knowledge_injection.data import convert_qa_to_tensors
from full_tofu_knowledge_injection.metrics import get_forget_quality, get_model_utility


def get_batch_loss(logits, labels):
    shifted = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_fn(logits.transpose(-1, -2), shifted).sum(dim=-1)


class TofuEvalDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int,
        tags: Dict[str, str],
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tags = tags
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[self.question_key]
        answers = row[self.answer_key]
        if isinstance(answers, str):
            answers = [answers]
        tensors = []
        for ans in answers:
            ids, labels, attn, _ = convert_qa_to_tensors(
                self.tokenizer,
                question,
                ans,
                self.max_length,
                self.tags["question_start_tag"],
                self.tags["question_end_tag"],
                self.tags["answer_tag"],
            )
            tensors.append((ids, labels, attn))
        if len(tensors) == 1:
            return tensors[0][0], tensors[0][1], tensors[0][2], torch.tensor(idx, dtype=torch.long)
        return (
            torch.stack([t[0] for t in tensors]),
            torch.stack([t[1] for t in tensors]),
            torch.stack([t[2] for t in tensors]),
            torch.tensor(idx, dtype=torch.long),
        )


def collate_eval(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    attn = torch.stack([b[2] for b in batch])
    indices = torch.stack([b[3] for b in batch])
    return input_ids, labels, attn, indices


def collate_eval_perturb(batch):
    """For perturbed answers which may be [n_pert, seq_len] per sample."""
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    attn = torch.stack([b[2] for b in batch])
    indices = torch.stack([b[3] for b in batch])
    return input_ids, labels, attn, indices


@torch.no_grad()
def run_generation(model, tokenizer, prompts: List[str], max_new_tokens: int = 64):
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        max_length=None,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


@torch.no_grad()
def eval_rouge(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1, rougeL = {}, {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        scores = scorer.score(gt, gen)
        rouge1[int(idx)] = scores["rouge1"].recall
        rougeL[int(idx)] = scores["rougeL"].recall
    return {"rouge1_recall": rouge1, "rougeL_recall": rougeL}


@torch.no_grad()
def eval_perturbation_ratio(model, base_loader, perturb_loader):
    logs = {
        "average_perturb_loss": {},
        "avg_paraphrased_loss": {},
        "truth_ratio": {},
        "paraphrased_loss": {},
        "perturb_loss": {},
        "num_token_paraphrased": {},
        "num_token_perturb": {},
    }
    for batch, perturb_batch in tqdm(
        zip(base_loader, perturb_loader), total=min(len(base_loader), len(perturb_loader))
    ):
        input_ids, labels, attn, indices = batch
        p_input_ids, p_labels, p_attn, _ = perturb_batch
        batch_d = {
            "input_ids": input_ids.to(model.device),
            "labels": labels.to(model.device),
            "attention_mask": attn.to(model.device),
        }
        # perturbed may be 2D [bsz, seq] (single) or 3D [bsz, n_pert, seq]
        if p_input_ids.dim() == 2:
            p_input_ids = p_input_ids.unsqueeze(1)
            p_labels = p_labels.unsqueeze(1)
            p_attn = p_attn.unsqueeze(1)
        bsz, n_pert, seq = p_input_ids.shape
        perturb_d = {
            "input_ids": p_input_ids.reshape(bsz * n_pert, seq).to(model.device),
            "labels": p_labels.reshape(bsz * n_pert, seq).to(model.device),
            "attention_mask": p_attn.reshape(bsz * n_pert, seq).to(model.device),
        }
        gt_loss = get_batch_loss(model(**batch_d).logits, batch_d["labels"])
        perturb_loss = get_batch_loss(model(**perturb_d).logits, perturb_d["labels"]).view(
            bsz, n_pert
        )
        num_gt = (batch_d["labels"] != -100).sum(-1)
        num_pert = (perturb_d["labels"] != -100).view(bsz, n_pert, seq).sum(-1)
        gt_per_token = gt_loss / num_gt
        perturb_per_token = perturb_loss / num_pert
        truth_ratio = torch.exp(gt_per_token - perturb_per_token.mean(-1))

        for i, idx in enumerate(indices.tolist()):
            logs["avg_paraphrased_loss"][idx] = float(gt_per_token[i])
            logs["average_perturb_loss"][idx] = perturb_per_token[i].float().cpu().numpy().tolist()
            logs["truth_ratio"][idx] = float(truth_ratio[i])
            logs["paraphrased_loss"][idx] = float(gt_loss[i])
            logs["perturb_loss"][idx] = perturb_loss[i].float().cpu().numpy().tolist()
            logs["num_token_paraphrased"][idx] = int(num_gt[i])
            logs["num_token_perturb"][idx] = num_pert[i].float().cpu().numpy().tolist()
    return logs


@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    tags: Dict[str, str],
    split: str,
    tofu_root: str,
    question_key: str,
    answer_key: str,
    base_answer_key: str,
    perturbed_answer_key: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
):
    data = load_tofu_split(tofu_root, split)
    indices = np.arange(len(data))
    data = data.add_column("index", indices.tolist())

    eval_ds = TofuEvalDataset(data, tokenizer, max_length, tags, question_key, answer_key)
    base_ds = TofuEvalDataset(data, tokenizer, max_length, tags, question_key, base_answer_key)
    perturb_ds = TofuEvalDataset(
        data, tokenizer, max_length, tags, question_key, perturbed_answer_key
    )

    eval_loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate_eval)
    base_loader = DataLoader(base_ds, batch_size=max(1, batch_size // 4), collate_fn=collate_eval)
    perturb_loader = DataLoader(
        perturb_ds, batch_size=max(1, batch_size // 4), collate_fn=collate_eval_perturb
    )

    logs: Dict = {}
    gen_outputs, ground_truths, all_indices = [], [], []

    for input_ids, labels, attn, indices in tqdm(eval_loader, desc=split):
        batch = {
            "input_ids": input_ids.to(model.device),
            "labels": labels.to(model.device),
            "attention_mask": attn.to(model.device),
        }
        outputs = model(**batch)
        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        num_gt = (batch["labels"] != -100).sum(-1)
        gt_per_token = gt_loss / num_gt

        prompts = []
        gts = []
        for i, idx in enumerate(indices.tolist()):
            q = data[int(idx)][question_key]
            gt = data[int(idx)][answer_key]
            if isinstance(gt, list):
                gt = gt[0]
            prompt = f"{tags['question_start_tag']}{q}{tags['question_end_tag']}{tags['answer_tag']}"
            prompts.append(prompt)
            gts.append(gt)

        gens = run_generation(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        gen_outputs.extend(gens)
        ground_truths.extend(gts)
        all_indices.extend(indices.tolist())

        for i, idx in enumerate(indices.tolist()):
            logs.setdefault("avg_gt_loss", {})[int(idx)] = float(gt_per_token[i].item())
            logs.setdefault("gt_loss", {})[int(idx)] = float(gt_loss[i].item())
            logs.setdefault("num_token_gt", {})[int(idx)] = int(num_gt[i].item())
            logs.setdefault("generated_text", {})[int(idx)] = (prompts[i], gens[i], gts[i])

    logs.update(eval_rouge(gen_outputs, ground_truths, all_indices))
    logs.update(eval_perturbation_ratio(model, base_loader, perturb_loader))
    return logs


EVAL_TASKS = [
    {
        "name": "eval_log.json",
        "split": "retain_perturbed",
        "answer_key": "answer",
        "base_answer_key": "paraphrased_answer",
        "perturbed_answer_key": "perturbed_answer",
    },
    {
        "name": "eval_real_author_wo_options.json",
        "split": "real_authors_perturbed",
        "answer_key": "answer",
        "base_answer_key": "answer",
        "perturbed_answer_key": "perturbed_answer",
    },
    {
        "name": "eval_real_world_wo_options.json",
        "split": "world_facts_perturbed",
        "answer_key": "answer",
        "base_answer_key": "answer",
        "perturbed_answer_key": "perturbed_answer",
    },
    {
        "name": "eval_log_forget.json",
        "split": "forget10_perturbed",
        "answer_key": "answer",
        "base_answer_key": "paraphrased_answer",
        "perturbed_answer_key": "perturbed_answer",
    },
]


def run_eval(model_path: str, cfg: dict, output_dir: Path, batch_size: int = 8):
    tags = {
        "question_start_tag": cfg["data"]["question_start_tag"],
        "question_end_tag": cfg["data"]["question_end_tag"],
        "answer_tag": cfg["data"]["answer_tag"],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    aggregated = {}
    for task in EVAL_TASKS:
        print(f"\n[eval] {task['name']} split={task['split']}")
        logs = evaluate_task(
            model,
            tokenizer,
            tags,
            task["split"],
            cfg["data"]["tofu_root"],
            "question",
            task["answer_key"],
            task["base_answer_key"],
            task["perturbed_answer_key"],
            batch_size=batch_size,
            max_length=int(cfg["data"]["max_length"]),
            max_new_tokens=64,
        )
        aggregated[task["name"]] = logs
        (output_dir / task["name"]).write_text(
            json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (output_dir / "eval_log_aggregated.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--full_model", default=str(ROOT / "outputs" / "full_tofu_merged"))
    parser.add_argument("--retain_model", default=str(ROOT / "outputs" / "retain90_merged"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--output_metrics",
        default=str(ROOT / "results" / "pre_unlearning_full_tofu_metrics.json"),
    )
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    full_eval_dir = ROOT / "results" / "eval_full_tofu"
    retain_eval_dir = ROOT / "results" / "eval_retain90_reference"
    full_eval_dir.mkdir(parents=True, exist_ok=True)
    retain_eval_dir.mkdir(parents=True, exist_ok=True)

    print("=== Evaluating θ_full ===")
    full_agg = run_eval(args.full_model, cfg, full_eval_dir, args.batch_size)
    print("=== Evaluating θ_retain (reference) ===")
    retain_agg = run_eval(args.retain_model, cfg, retain_eval_dir, args.batch_size)

    utility = get_model_utility(full_agg)
    fq = get_forget_quality(full_agg, retain_agg)

    report = {
        "model_full": args.full_model,
        "model_retain_reference": args.retain_model,
        "model_utility": utility,
        "forget_quality": fq,
        "per_task_full": {
            k: {
                "n_samples": len(v.get("rougeL_recall", {})),
                "rougeL_mean": float(np.mean(list(v["rougeL_recall"].values()))),
                "avg_gt_prob_mean": float(
                    np.mean(np.exp(-1 * np.array(list(v["avg_gt_loss"].values()))))
                ),
            }
            for k, v in full_agg.items()
        },
    }
    out = Path(args.output_metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # save truth ratio distributions
    (ROOT / "results" / "full_tofu_forget10_truth_ratios.json").write_text(
        json.dumps(
            {
                "full": fq["full_truth_ratio_per_sample"],
                "retain_reference": fq["retain_truth_ratio_per_sample"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== TOFU Pre-Unlearning Metrics ===")
    print(f"Model Utility: {utility['Model Utility']:.4f}")
    print(f"Forget Quality p-value: {fq['Forget Quality']:.6e}")
    print(f"KS statistic: {fq['KS Test Forget']:.4f}")
    print(f"[✓] wrote {out}")


if __name__ == "__main__":
    main()
