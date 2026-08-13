#!/usr/bin/env python3
"""Step 12: QA sanity check on forget10 / retain90 (θ0 vs θ_full)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection import load_tofu_split, load_yaml_config


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(pred: str, ref: str) -> bool:
    return normalize(pred) == normalize(ref)


def rouge_l(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(ref, pred)["rougeL"].fmeasure)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def answer_log_likelihood(
    model, tokenizer, prompt: str, answer: str, max_length: int = 512
) -> float:
    """Mean log-prob per ground-truth answer token (higher = better)."""
    full = prompt + answer
    enc = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    start = max(prompt_len - 1, 0)
    answer_lp = token_lp[start:]
    if answer_lp.numel() == 0:
        return float("-inf")
    return float(answer_lp.mean().item())


def build_gen_prompt(question: str, tags: Dict[str, str]) -> str:
    return (
        f"{tags['question_start_tag']}{question}"
        f"{tags['question_end_tag']}{tags['answer_tag']}"
    )


def run_model_on_split(
    model_path: str,
    split_name: str,
    indices: List[int],
    ds,
    tags: Dict[str, str],
    max_new_tokens: int,
) -> List[dict]:
    print(f"[load] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = []
    for j, idx in enumerate(indices):
        q = ds[idx]["question"]
        gt = ds[idx]["answer"]
        prompt = build_gen_prompt(q, tags)
        pred = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        gt_lp = answer_log_likelihood(model, tokenizer, prompt, gt)
        rows.append(
            {
                "index": int(idx),
                "question": q,
                "ground_truth": gt,
                "answer": pred,
                "gt_log_likelihood": gt_lp,
            }
        )
        if (j + 1) % 10 == 0 or j + 1 == len(indices):
            print(f"  [{split_name}] {j + 1}/{len(indices)} done")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(name: str, base_rows: List[dict], full_rows: List[dict]) -> dict:
    assert len(base_rows) == len(full_rows)
    n = len(base_rows)
    base_em = base_rl = full_em = full_rl = 0.0
    base_lp = full_lp = 0.0
    full_wins_em = full_wins_lp = 0
    examples = []

    for b, f in zip(base_rows, full_rows):
        gt = b["ground_truth"]
        bem = exact_match(b["answer"], gt)
        fem = exact_match(f["answer"], gt)
        brl = rouge_l(b["answer"], gt)
        frl = rouge_l(f["answer"], gt)
        base_em += int(bem)
        full_em += int(fem)
        base_rl += brl
        full_rl += frl
        base_lp += b["gt_log_likelihood"]
        full_lp += f["gt_log_likelihood"]
        full_wins_em += int(fem and not bem)
        full_wins_lp += int(f["gt_log_likelihood"] > b["gt_log_likelihood"])

        if len(examples) < 5:
            examples.append(
                {
                    "question": b["question"],
                    "ground_truth": gt,
                    "base_answer": b["answer"],
                    "full_answer": f["answer"],
                    "base_exact_match": bem,
                    "full_exact_match": fem,
                    "base_rouge_l": brl,
                    "full_rouge_l": frl,
                    "base_gt_log_likelihood": b["gt_log_likelihood"],
                    "full_gt_log_likelihood": f["gt_log_likelihood"],
                }
            )

    return {
        "split": name,
        "n_samples": n,
        "base_exact_match": base_em / n,
        "full_exact_match": full_em / n,
        "exact_match_delta": (full_em - base_em) / n,
        "base_rouge_l": base_rl / n,
        "full_rouge_l": full_rl / n,
        "rouge_l_delta": (full_rl - base_rl) / n,
        "base_mean_gt_log_likelihood": base_lp / n,
        "full_mean_gt_log_likelihood": full_lp / n,
        "gt_log_likelihood_delta": (full_lp - base_lp) / n,
        "full_wins_exact_match": full_wins_em,
        "full_wins_gt_log_likelihood": full_wins_lp,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--merged_path", default=str(ROOT / "outputs" / "full_tofu_merged"))
    parser.add_argument("--n_per_split", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "pre_unlearning_qa_sanity.json"),
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    tags = {
        "question_start_tag": cfg["data"]["question_start_tag"],
        "question_end_tag": cfg["data"]["question_end_tag"],
        "answer_tag": cfg["data"]["answer_tag"],
    }
    base_path = cfg["model"]["path"]
    tofu_root = cfg["data"]["tofu_root"]

    report = {
        "base_model": base_path,
        "full_model": args.merged_path,
        "n_per_split": args.n_per_split,
        "seed": args.seed,
        "prompt_template": tags,
        "splits": {},
    }

    for split_name in ["forget10", "retain90"]:
        ds = load_tofu_split(tofu_root, split_name)
        n = min(args.n_per_split, len(ds))
        g = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(ds), generator=g)[:n].tolist()
        print(f"\n=== {split_name}: {n} samples ===")

        base_rows = run_model_on_split(
            base_path, split_name, indices, ds, tags, args.max_new_tokens
        )
        full_rows = run_model_on_split(
            args.merged_path, split_name, indices, ds, tags, args.max_new_tokens
        )
        report["splits"][split_name] = summarize(split_name, base_rows, full_rows)
        report["splits"][split_name]["sample_indices"] = indices
        report["splits"][split_name]["all_samples"] = [
            {
                "index": b["index"],
                "question": b["question"],
                "ground_truth": b["ground_truth"],
                "base_answer": b["answer"],
                "full_answer": f["answer"],
                "base_exact_match": exact_match(b["answer"], b["ground_truth"]),
                "full_exact_match": exact_match(f["answer"], f["ground_truth"]),
                "base_rouge_l": rouge_l(b["answer"], b["ground_truth"]),
                "full_rouge_l": rouge_l(f["answer"], f["ground_truth"]),
                "base_gt_log_likelihood": b["gt_log_likelihood"],
                "full_gt_log_likelihood": f["gt_log_likelihood"],
            }
            for b, f in zip(base_rows, full_rows)
        ]

    f10 = report["splits"]["forget10"]
    r90 = report["splits"]["retain90"]
    report["knowledge_learned"] = {
        "forget10_full_better_em": f10["full_exact_match"] > f10["base_exact_match"],
        "retain90_full_better_em": r90["full_exact_match"] > r90["base_exact_match"],
        "forget10_full_better_rouge": f10["full_rouge_l"] > f10["base_rouge_l"],
        "retain90_full_better_rouge": r90["full_rouge_l"] > r90["base_rouge_l"],
        "forget10_full_better_lp": f10["full_mean_gt_log_likelihood"] > f10["base_mean_gt_log_likelihood"],
        "retain90_full_better_lp": r90["full_mean_gt_log_likelihood"] > r90["base_mean_gt_log_likelihood"],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[✓] wrote {out}")
    for split_name in ["forget10", "retain90"]:
        s = report["splits"][split_name]
        print(
            f"[{split_name}] EM: base={s['base_exact_match']:.3f} full={s['full_exact_match']:.3f} | "
            f"ROUGE-L: base={s['base_rouge_l']:.3f} full={s['full_rouge_l']:.3f} | "
            f"GT log-lik: base={s['base_mean_gt_log_likelihood']:.3f} full={s['full_mean_gt_log_likelihood']:.3f}"
        )


if __name__ == "__main__":
    main()
