"""Compare actual continuous and resumed CPU training through the lab launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def differences(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        return [] if (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                      and left.shape == right.shape and torch.equal(left, right)) else [path]
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ".keys"]
        return [p for key in left for p in differences(left[key], right[key], f"{path}.{key}")]
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [path + ".length"]
        return [p for i, (a, b) in enumerate(zip(left, right))
                for p in differences(a, b, f"{path}[{i}]")]
    return [] if left == right else [path]


def scientific_evaluation(value):
    # Only runtime measurements and the evaluation phase can differ by design.
    omitted = {"wall_seconds", "collection_wall_seconds", "process_peak_rss_mib", "phase"}
    if isinstance(value, dict):
        return {key: scientific_evaluation(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [scientific_evaluation(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    args = parser.parse_args()
    left, right = [torch.load(path, weights_only=True, map_location="cpu")
                   for path in (args.left, args.right)]
    fields = ("schema", "scope", "model", "optimizer", "completed_episodes",
              "next_seed_index", "rng", "configs", "versions")
    checks = {key: differences(left[key], right[key], key) for key in fields}
    checks["evaluation"] = differences(scientific_evaluation(left["evaluation"]),
                                       scientific_evaluation(right["evaluation"]), "evaluation")
    best_left, best_right = left["best_checkpoint"], right["best_checkpoint"]
    if (best_left is None) != (best_right is None):
        checks["best_checkpoint"] = ["best_checkpoint presence"]
    elif best_left is not None:
        checks["best_checkpoint"] = [p for key in fields
            for p in differences(best_left[key], best_right[key], "best_checkpoint." + key)]
        checks["best_evaluation"] = differences(scientific_evaluation(best_left["evaluation"]),
            scientific_evaluation(best_right["evaluation"]), "best_evaluation")
    result = {"left": args.left, "right": args.right, "completed_episodes": left["completed_episodes"],
              "all_equal": not any(checks.values()), "differences": checks,
              "comparison": "Exact tensor/scalar equality. Evaluation phase and measured runtime/RSS excluded; full configs remain exact."}
    (Path(os.environ["LAB_RUN_DIR"]) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return int(not result["all_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
