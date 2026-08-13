#!/usr/bin/env python3
"""Validate LoRA adapter vs merged θ_full equivalence (greedy decode + logits)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection import load_tofu_split, load_yaml_config
from full_tofu_knowledge_injection.data import build_prompt_answer


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def next_token_logits(model, tokenizer, text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1, :].float().cpu()
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "merge_validation.json"),
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    base_path = cfg["model"]["path"]
    adapter_dir = Path(cfg["paths"]["adapter_dir"])
    merged_dir = ROOT / "outputs" / "full_tofu_merged"

    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    if not merged_dir.exists():
        raise FileNotFoundError(merged_dir)

    tags = {
        "question_start_tag": cfg["data"]["question_start_tag"],
        "question_end_tag": cfg["data"]["question_end_tag"],
        "answer_tag": cfg["data"]["answer_tag"],
    }

    ds = load_tofu_split(cfg["data"]["tofu_root"], "full")
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=rng)[: args.n_samples].tolist()

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[A] loading base+adapter on GPU...")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="auto"
    )
    adapter_model = PeftModel.from_pretrained(adapter_model, str(adapter_dir))
    adapter_model.eval()

    print("[B] loading merged θ_full on GPU (after freeing A temporarily via CPU offload if needed)...")
    # Keep both on GPU if memory allows (~2x 16GB bf16 weights + activations). A6000 48GB usually OK for inference-only.
    merged_model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir), dtype=torch.bfloat16, device_map="auto"
    )
    merged_model.eval()

    rows: List[dict] = []
    text_mismatch = 0
    logit_max_abs_all = []
    logit_mean_abs_all = []

    for i in idx:
        q = ds[int(i)]["question"]
        a = ds[int(i)]["answer"]
        prompt, _, _ = build_prompt_answer(q, a, **tags)
        # generation prompt: question side only + answer tag (no gold answer)
        gen_prompt = f"{tags['question_start_tag']}{q}{tags['question_end_tag']}{tags['answer_tag']}"

        out_a = greedy_generate(adapter_model, tokenizer, gen_prompt, args.max_new_tokens)
        out_b = greedy_generate(merged_model, tokenizer, gen_prompt, args.max_new_tokens)
        la = next_token_logits(adapter_model, tokenizer, gen_prompt)
        lb = next_token_logits(merged_model, tokenizer, gen_prompt)
        diff = (la - lb).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        logit_max_abs_all.append(max_abs)
        logit_mean_abs_all.append(mean_abs)
        same_text = out_a.strip() == out_b.strip()
        if not same_text:
            text_mismatch += 1
        # Loose pass: exact text match OR very small logit drift
        sample_pass = same_text or (max_abs < 1e-2 and mean_abs < 1e-3)
        rows.append(
            {
                "index": int(i),
                "question": q,
                "ground_truth": a,
                "adapter_output": out_a,
                "merged_output": out_b,
                "text_exact_match": same_text,
                "logit_max_abs": max_abs,
                "logit_mean_abs": mean_abs,
                "pass": bool(sample_pass),
            }
        )
        print(
            f"[{len(rows)}/{args.n_samples}] idx={i} text_match={same_text} "
            f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} pass={sample_pass}"
        )

    n = len(rows)
    pass_rate = sum(1 for r in rows if r["pass"]) / n
    overall_pass = pass_rate >= 0.9 and (sum(logit_max_abs_all) / n) < 0.05
    report = {
        "n_samples": n,
        "text_mismatch": text_mismatch,
        "text_exact_match_rate": 1.0 - text_mismatch / n,
        "logit_max_abs_mean": float(sum(logit_max_abs_all) / n),
        "logit_max_abs_max": float(max(logit_max_abs_all)),
        "logit_mean_abs_mean": float(sum(logit_mean_abs_all) / n),
        "pass_rate": pass_rate,
        "overall": "PASS" if overall_pass else "FAIL",
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "samples": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] wrote {out_path}")
    print(f"[✓] merge validation: {report['overall']} (pass_rate={pass_rate:.3f})")


if __name__ == "__main__":
    main()
