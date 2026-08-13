#!/usr/bin/env python3
"""Merge knowledge-injection LoRA into base Llama-3-8B (θ_full or θ_retain)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    base_path = cfg["model"]["path"]
    adapter_dir = Path(args.adapter_dir or cfg["paths"]["adapter_dir"])
    default_merged = (
        ROOT / "outputs" / "retain90_merged"
        if "retain90" in cfg.get("experiment_name", "")
        else ROOT / "outputs" / "full_tofu_merged"
    )
    output_dir = Path(
        args.output_dir
        or cfg["paths"].get("merged_dir")
        or default_merged
    )

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
    required = ["adapter_config.json"]
    for name in required:
        if not (adapter_dir / name).exists():
            raise FileNotFoundError(f"Missing {name} under {adapter_dir}")
    weight_ok = (adapter_dir / "adapter_model.safetensors").exists() or (
        adapter_dir / "adapter_model.bin"
    ).exists()
    if not weight_ok:
        raise FileNotFoundError(f"No adapter weights under {adapter_dir}")

    print(f"[merge] base={base_path}")
    print(f"[merge] adapter={adapter_dir}")
    print(f"[merge] output={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=dtype,
        device_map="cpu",  # merge on CPU to avoid GPU OOM; then save
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] merging...")
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # ensure generation_config exists if present on base
    gen_src = Path(base_path) / "generation_config.json"
    if gen_src.exists() and not (output_dir / "generation_config.json").exists():
        (output_dir / "generation_config.json").write_text(
            gen_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": base_path,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(output_dir),
        "dtype": "bfloat16",
        "files": sorted(p.name for p in output_dir.iterdir()),
    }
    (output_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[✓] merged θ_full saved ->", output_dir)
    print("[✓] files:", meta["files"])


if __name__ == "__main__":
    main()
