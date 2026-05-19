"""
Main pipeline:
  1. Use train samples to build the metric database (cluster + generate metrics).
  2. Use test samples to judge A/B/tie with the metric database.

Arguments:
  --skip-build    Skip database construction and use existing metrics.
  --build-only    Build the database only; do not evaluate.
  --train-num     Number of samples for database construction (default: 300).
  --test-num      Number of samples for evaluation (default: 200).
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from datasets import load_from_disk

from cluster_by_dimensions import (
    DATASET_PATH,
    DIMENSIONS,
    OUTPUT_DIR as CLUSTER_DIR,
    cluster_one_dimension,
    compute_group_bounds,
    extract_task,
)
from config import JUDGMENT_OUTPUT_DIR, SEED, ensure_directories
from generate_metrics import OUTPUT_DIR as METRICS_DIR
from generate_metrics import process_single_group
from judge_with_metrics import judge_sample, load_metrics_index


JUDGMENT_DIR = JUDGMENT_OUTPUT_DIR
MAX_WORKERS_BUILD = 4
MAX_WORKERS_EVAL = 10
ensure_directories()


def load_dataset():
    """Load the full dataset from disk."""
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    data = dataset["train"]
    print(f"  Dataset size: {len(data)} rows\n")
    return data


def _is_non_tie_winner(winner):
    return "model_a" in winner or "model_b" in winner


def sample_split(data, train_num, test_num):
    """Sample non-overlapping train/test rows from the dataset."""
    eligible_indices = [
        idx for idx in range(len(data))
        if _is_non_tie_winner(data[idx]["winner"])
    ]
    total = len(eligible_indices)
    need = train_num + test_num
    if need > total:
        print(f"  Need {need} rows but only have {total}; scaling down.")
        train_num = int(total * train_num / need)
        test_num = total - train_num
        need = train_num + test_num

    random.seed(SEED)
    all_indices = random.sample(eligible_indices, need)
    train_indices = all_indices[:train_num]
    test_indices = all_indices[train_num : train_num + test_num]

    def to_samples(indices):
        samples = []
        for idx in indices:
            row = data[idx]
            if not _is_non_tie_winner(row["winner"]):
                continue
            samples.append(
                {
                    "index": idx,
                    "question_id": row["question_id"],
                    "task": extract_task(row["conversation_a"]),
                    "conversation_a": row["conversation_a"],
                    "conversation_b": row["conversation_b"],
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "winner": row["winner"],
                }
            )
        return samples

    train_samples = to_samples(train_indices)
    test_samples = to_samples(test_indices)

    print(f"  Train set: {len(train_samples)} rows (cluster + metric generation)")
    print(f"  Test set:  {len(test_samples)} rows (judgment)")
    print("  Non-overlapping split: ok\n")

    return train_samples, test_samples


def build_database(train_samples):
    """Phase 1: cluster samples, then generate metrics for each valid group."""
    print("=" * 70)
    print("  Phase 1: Build Metric Database")
    print("=" * 70)

    phase1_start = time.time()

    print("\n" + "-" * 50)
    print("  Step 1.1: Five-dimensional clustering")
    print("-" * 50)

    cluster_results = {}

    for dim_config in DIMENSIONS:
        dim_id = dim_config["id"]
        labels = cluster_one_dimension(dim_config, train_samples)
        cluster_results[dim_id] = labels

        counts = Counter(labels)
        groups = {}
        output_samples = []

        for i, sample in enumerate(train_samples):
            label = labels[i]
            output_samples.append(
                {
                    **sample,
                    f"cluster_{dim_id}": label,
                }
            )
            if label not in groups:
                groups[label] = {"count": 0, "question_ids": []}
            groups[label]["count"] += 1
            groups[label]["question_ids"].append(sample["question_id"])

        lower, upper = compute_group_bounds(len(train_samples), len(dim_config["labels"]))

        for group in groups.values():
            group["within_bounds"] = lower <= group["count"] <= upper

        cluster_data = {
            "dimension_id": dim_id,
            "dimension_name": dim_config["name"],
            "description": dim_config["description"],
            "available_labels": dim_config["labels"],
            "granularity": {
                "N": len(train_samples),
                "K": len(dim_config["labels"]),
                "lower_bound": lower,
                "upper_bound": upper,
            },
            "label_distribution": dict(counts.most_common()),
            "num_groups": len(counts),
            "num_samples": len(train_samples),
            "groups": groups,
            "samples": output_samples,
        }

        os.makedirs(CLUSTER_DIR, exist_ok=True)
        out_file = os.path.join(CLUSTER_DIR, f"cluster_{dim_id}.json")
        with open(out_file, "w", encoding="utf-8") as file:
            json.dump(cluster_data, file, ensure_ascii=False, indent=2)
        print(f"  Saved {out_file}")

    print("\n" + "-" * 50)
    print("  Step 1.2: Generate Metrics")
    print("-" * 50)

    total_metrics = 0
    total_groups = 0
    group_jobs = []

    for dim_config in DIMENSIONS:
        dim_id = dim_config["id"]
        cluster_file = os.path.join(CLUSTER_DIR, f"cluster_{dim_id}.json")

        with open(cluster_file, "r", encoding="utf-8") as file:
            cluster_data = json.load(file)

        groups = cluster_data["groups"]

        for group_label, group_info in groups.items():
            count = group_info["count"]
            if count < 6:
                print(f"  Skip {dim_id}/{group_label} ({count} rows; too small)")
                continue

            total_groups += 1
            group_jobs.append((dim_id, cluster_file, group_label, count))

    def _process_group(job):
        dim_id, cluster_file, group_label, count = job
        print(f"\n  Processing {dim_id} / {group_label} ({count} rows)")
        result = process_single_group(cluster_file, group_label)
        return dim_id, group_label, result

    os.makedirs(METRICS_DIR, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_BUILD) as executor:
        future_to_job = {
            executor.submit(_process_group, job): job
            for job in group_jobs
        }
        for future in as_completed(future_to_job):
            dim_id, group_label, result = future.result()

            if result:
                num_metrics = result["num_verified_metrics"]
                total_metrics += num_metrics

                safe_label = group_label.replace("/", "_").replace(" ", "_")
                out_file = os.path.join(METRICS_DIR, f"metrics_{dim_id}_{safe_label}.json")
                with open(out_file, "w", encoding="utf-8") as file:
                    json.dump(result, file, ensure_ascii=False, indent=2)
                print(f"    Saved {num_metrics} metrics -> {out_file}")

    phase1_elapsed = time.time() - phase1_start

    print(f"\n{'-' * 50}")
    print("  Phase 1 complete")
    print(f"  Groups processed: {total_groups}")
    print(f"  Metrics generated: {total_metrics}")
    print(f"  Elapsed: {phase1_elapsed:.1f}s")
    print(f"{'-' * 50}\n")

    return total_groups, total_metrics


def run_evaluation(test_samples):
    """Phase 2: judge test samples one by one."""
    print("=" * 70)
    print("  Phase 2: Judge Test Samples")
    print("=" * 70)

    import judge_with_metrics

    judge_with_metrics._metrics_cache = None
    index = load_metrics_index()
    total_metrics = sum(len(value) for value in index.values())
    print(f"  Loaded metric database: {len(index)} groups, {total_metrics} metrics\n")

    phase2_start = time.time()
    results = []
    results_by_index = [None] * len(test_samples)
    correct = 0
    total = 0

    counter_lock = Lock()

    def _evaluate_one(i, sample):
        question_id = sample.get("question_id", f"sample_{i}")
        task_short = sample["task"][:80].replace("\n", " ")

        result = judge_sample(sample)

        gt_raw = sample["winner"]
        if "model_a" in gt_raw:
            ground_truth = "A"
        elif "model_b" in gt_raw:
            ground_truth = "B"
        else:
            ground_truth = "tie"

        verdict = result["verdict"]
        is_correct = verdict == ground_truth
        return i, question_id, task_short, {
            "question_id": question_id,
            "task": sample["task"][:300],
            "model_a": sample.get("model_a", ""),
            "model_b": sample.get("model_b", ""),
            "labels": result["labels"],
            "metrics_count": len(result["metrics_used"]),
            "metrics_used": result["metrics_used"],
            "verdict": verdict,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "reasoning": result["reasoning"],
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EVAL) as executor:
        future_to_index = {
            executor.submit(_evaluate_one, i, sample): i
            for i, sample in enumerate(test_samples)
        }
        for future in as_completed(future_to_index):
            i, question_id, task_short, result_row = future.result()
            results_by_index[i] = result_row

            with counter_lock:
                total += 1
                if result_row["correct"]:
                    correct += 1
                accuracy_so_far = correct / total

            mark = "ok" if result_row["correct"] else "wrong"
            print(
                f"  [{i + 1}/{len(test_samples)}] {question_id}: {task_short}..."
                f"  -> {result_row['verdict']} (GT={result_row['ground_truth']}) "
                f"{mark} [{correct}/{total}={accuracy_so_far:.1%}]"
            )

            if total % 50 == 0:
                partial_results = [row for row in results_by_index if row is not None]
                _save_judgments(partial_results, correct, total, "judgment_results.json")

    phase2_elapsed = time.time() - phase2_start
    results = [row for row in results_by_index if row is not None]
    _save_judgments(results, correct, total, "judgment_results.json")

    return results, correct, total, phase2_elapsed


def _save_judgments(results, correct, total, filename):
    """Save judgment results and summary statistics."""
    accuracy = correct / total if total > 0 else 0

    label_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        for dim, label in result["labels"].items():
            key = f"{dim}={label}"
            label_stats[key]["total"] += 1
            if result["correct"]:
                label_stats[key]["correct"] += 1

    metric_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        metrics_count = result["metrics_count"]
        bucket = str(metrics_count) if metrics_count <= 10 else "10+"
        metric_stats[bucket]["total"] += 1
        if result["correct"]:
            metric_stats[bucket]["correct"] += 1

    output = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
        },
        "accuracy_by_metrics_count": {
            key: {
                **value,
                "accuracy": round(value["correct"] / value["total"], 4) if value["total"] else 0,
            }
            for key, value in sorted(metric_stats.items())
        },
        "accuracy_by_label": {
            key: {
                **value,
                "accuracy": round(value["correct"] / value["total"], 4) if value["total"] else 0,
            }
            for key, value in sorted(label_stats.items())
            if value["total"] >= 3
        },
        "judgments": results,
    }

    os.makedirs(JUDGMENT_DIR, exist_ok=True)
    out_path = os.path.join(JUDGMENT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def print_summary(
    results,
    correct,
    total,
    phase2_elapsed,
    train_num,
    test_num,
    total_groups,
    total_metrics,
    build_time=None,
):
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'=' * 70}")
    print("  Final Summary")
    print(f"{'=' * 70}")

    if build_time is not None:
        print("\n  [Phase 1 - Build]")
        print(f"    Train samples:      {train_num}")
        print(f"    Cluster groups:     {total_groups}")
        print(f"    Generated metrics:  {total_metrics}")
        print(f"    Elapsed:            {build_time:.1f}s")

    print("\n  [Phase 2 - Judgment]")
    print(f"    Test samples:       {test_num}")
    print(f"    Correct:            {correct}")
    print(f"    Total:              {total}")
    print(f"    Accuracy:           {accuracy:.1%}")
    print(f"    Elapsed:            {phase2_elapsed:.1f}s")
    if total:
        print(f"    Per sample:         {phase2_elapsed / total:.1f}s")

    verdict_counts = Counter(result["verdict"] for result in results)
    gt_counts = Counter(result["ground_truth"] for result in results)
    print("\n  [Verdict Distribution]")
    print(f"    {'':10s} {'Pred':>6s}  {'GT':>6s}")
    for verdict in ["A", "B"]:
        print(f"    {verdict:10s} {verdict_counts.get(verdict, 0):>6d}  {gt_counts.get(verdict, 0):>6d}")

    gt_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        gt_acc[result["ground_truth"]]["total"] += 1
        if result["correct"]:
            gt_acc[result["ground_truth"]]["correct"] += 1
    print("\n  [Accuracy by GT]")
    for verdict in ["A", "B"]:
        if gt_acc[verdict]["total"] > 0:
            group_accuracy = gt_acc[verdict]["correct"] / gt_acc[verdict]["total"]
            print(
                f"    GT={verdict:4s}: "
                f"{gt_acc[verdict]['correct']}/{gt_acc[verdict]['total']} = {group_accuracy:.1%}"
            )

    with_metrics = [result for result in results if result["metrics_count"] > 0]
    without_metrics = [result for result in results if result["metrics_count"] == 0]
    if with_metrics:
        acc_with = sum(result["correct"] for result in with_metrics) / len(with_metrics)
        print(
            f"\n  [With metric hits]    "
            f"{sum(result['correct'] for result in with_metrics)}/{len(with_metrics)} = {acc_with:.1%}"
        )
    if without_metrics:
        acc_without = sum(result["correct"] for result in without_metrics) / len(without_metrics)
        print(
            f"  [Without metric hits] "
            f"{sum(result['correct'] for result in without_metrics)}/{len(without_metrics)} = {acc_without:.1%}"
        )

    out_path = os.path.join(JUDGMENT_DIR, "judgment_results.json")
    print(f"\n  Results file: {out_path}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Build a metric database and judge Chatbot Arena samples.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip database construction and use existing metrics.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the database only; do not evaluate.",
    )
    parser.add_argument(
        "--train-num",
        type=int,
        default=200,
        help="Number of samples for database construction (default: 300).",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=200,
        help="Number of samples for evaluation (default: 200).",
    )
    args = parser.parse_args()

    overall_start = time.time()

    data = load_dataset()
    train_samples, test_samples = sample_split(data, args.train_num, args.test_num)

    total_groups = 0
    total_metrics = 0
    build_time = None

    if not args.skip_build:
        build_start = time.time()
        total_groups, total_metrics = build_database(train_samples)
        build_time = time.time() - build_start
    else:
        print("Skip Phase 1 (--skip-build); using existing metrics.\n")

    if args.build_only:
        print("Build-only mode (--build-only); skipping evaluation.")
        print(f"Total elapsed: {time.time() - overall_start:.1f}s")
        return

    results, correct, total, phase2_elapsed = run_evaluation(test_samples)

    print_summary(
        results,
        correct,
        total,
        phase2_elapsed,
        args.train_num,
        args.test_num,
        total_groups,
        total_metrics,
        build_time,
    )

    print(f"Total elapsed: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
