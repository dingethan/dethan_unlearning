#!/usr/bin/env python3
"""Lightweight TOFU eval: Retain + Forget only (for LR smoke screening).

Writes metrics under results/model_utility_optimization/.
Does NOT touch baseline outputs/full_tofu_merged.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection.config import load_yaml_config
from full_tofu_knowledge_injection.metrics import get_forget_quality, get_model_utility
from scripts.evaluate_tofu import EVAL_TASKS, evaluate_task


LIGHT_TASKS = [t for t in EVAL_TASKS if t["name"] in ("eval_log.json", "eval_log_forget.json")]
# Still need Real Authors / Real World for true Model Utility; for smoke we report retain/forget
# and a partial harmonic mean over available non-forget metrics if full tasks requested.


def run_light_eval(model_path: str, cfg: dict, output_dir: Path, batch_size: int, full_utility: bool):
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

    tasks = EVAL_TASKS if full_utility else LIGHT_TASKS
    aggregated = {}
    for task in tasks:
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

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return aggregated


def summarize_split(logs: dict) -> dict:
    rouge = float(np.mean(list(logs["rougeL_recall"].values())))
    prob = float(np.mean(np.exp(-1 * np.array(list(logs["avg_gt_loss"].values())))))
    para = np.array(list(logs["avg_paraphrased_loss"].values()))
    pert = np.array(list(logs["average_perturb_loss"].values())).mean(axis=-1)
    tr = np.exp(pert - para)
    # retain-style truth ratio display (same as metrics for non-forget)
    truth_ratio = float(np.mean(np.maximum(0, 1 - 1 / tr)))
    return {"ROUGE": rouge, "Prob": prob, "Truth_Ratio": truth_ratio, "truth_ratio_raw_mean": float(tr.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--full_utility", action="store_true")
    parser.add_argument(
        "--retain_agg",
        default=str(ROOT / "results" / "eval_retain90_reference" / "eval_log_aggregated.json"),
    )
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    out_dir = ROOT / "results" / "model_utility_optimization" / "evals" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Light eval run_id={args.run_id} model={args.model} ===")
    agg = run_light_eval(args.model, cfg, out_dir, args.batch_size, args.full_utility)

    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "Retain": summarize_split(agg["eval_log.json"]),
        "Forget": summarize_split(agg["eval_log_forget.json"]),
    }

    if args.full_utility and all(
        k in agg
        for k in (
            "eval_log.json",
            "eval_real_author_wo_options.json",
            "eval_real_world_wo_options.json",
            "eval_log_forget.json",
        )
    ):
        summary["model_utility"] = get_model_utility(agg)
        if Path(args.retain_agg).exists():
            retain_agg = json.loads(Path(args.retain_agg).read_text(encoding="utf-8"))
            # need forget task in retain_agg
            summary["forget_quality"] = get_forget_quality(agg, retain_agg)

    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[✓] wrote {out}")


if __name__ == "__main__":
    main()
