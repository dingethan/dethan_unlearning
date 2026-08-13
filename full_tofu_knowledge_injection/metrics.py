"""Official TOFU benchmark metrics (locuslab/tofu)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import hmean, ks_2samp


def get_model_utility(eval_result_dict: Dict) -> Dict[str, float]:
    """Harmonic mean over Retain / Real Authors / Real World metrics (excludes Forget)."""
    eval_task_dict = {
        "eval_real_author_wo_options.json": "Real Authors",
        "eval_real_world_wo_options.json": "Real World",
        "eval_log.json": "Retain",
        "eval_log_forget.json": "Forget",
    }
    output_result: Dict[str, float] = {}

    for k, v in eval_result_dict.items():
        if k not in eval_task_dict:
            continue
        name = eval_task_dict[k]

        if "eval_log" in k:
            gt_probs = np.exp(-1 * np.array(list(v["avg_gt_loss"].values())))
            avg_gt_prob = float(np.mean(gt_probs))
        else:
            avg_true_prob = np.exp(-1 * np.array(list(v["avg_gt_loss"].values())))
            avg_false_prob = np.exp(-1 * np.array(list(v["average_perturb_loss"].values())))
            avg_all_prob = np.concatenate(
                [np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1
            ).sum(-1)
            avg_gt_prob = float(np.mean(avg_true_prob / avg_all_prob))
        output_result[f"Prob. {name}"] = avg_gt_prob

        avg_rouge = float(np.array(list(v["rougeL_recall"].values())).mean())
        output_result[f"ROUGE {name}"] = avg_rouge

        avg_paraphrase = np.array(list(v["avg_paraphrased_loss"].values()))
        avg_perturbed = np.array(list(v["average_perturb_loss"].values()))
        avg_perturbed = avg_perturbed.mean(axis=-1)
        curr_stat = np.exp(avg_perturbed - avg_paraphrase)
        if "forget" in k:
            truth_ratio = float(np.mean(np.minimum(curr_stat, 1 / curr_stat)))
        else:
            truth_ratio = float(np.mean(np.maximum(0, 1 - 1 / curr_stat)))
        output_result[f"Truth Ratio {name}"] = truth_ratio

    model_utility_cands = [v for key, v in output_result.items() if "Forget" not in key]
    output_result["Model Utility"] = float(hmean(model_utility_cands))
    return output_result


def get_forget_quality(unlearn_result: Dict, retain_result: Dict) -> Dict[str, float]:
    """KS test on forget10 Truth Ratio: θ_full vs retain-only reference."""
    unlearn_forget = unlearn_result["eval_log_forget.json"]
    retain_forget = retain_result["eval_log_forget.json"]

    unlearn_para = np.array(list(unlearn_forget["avg_paraphrased_loss"].values()))
    unlearn_pert = np.array(list(unlearn_forget["average_perturb_loss"].values())).mean(axis=-1)
    retain_para = np.array(list(retain_forget["avg_paraphrased_loss"].values()))
    retain_pert = np.array(list(retain_forget["average_perturb_loss"].values())).mean(axis=-1)

    unlearn_truth_ratio = np.exp(unlearn_pert - unlearn_para)
    retain_truth_ratio = np.exp(retain_pert - retain_para)
    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {
        "Forget Quality": float(test_res.pvalue),
        "KS Test PVal Forget": float(test_res.pvalue),
        "KS Test Forget": float(test_res.statistic),
        "full_truth_ratio_mean": float(unlearn_truth_ratio.mean()),
        "retain_truth_ratio_mean": float(retain_truth_ratio.mean()),
        "full_truth_ratio_std": float(unlearn_truth_ratio.std()),
        "retain_truth_ratio_std": float(retain_truth_ratio.std()),
        "full_truth_ratio_per_sample": unlearn_truth_ratio.tolist(),
        "retain_truth_ratio_per_sample": retain_truth_ratio.tolist(),
    }
