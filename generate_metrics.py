"""
Generate and verify evaluation metrics for each cluster group.

Flow per group:
  1. Sample tasks from the group
  2. Ask LLM to propose candidate metrics
  3. For each metric, test it on sample A/B pairs to compute accuracy
  4. Keep metrics above accuracy threshold
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    METRICS_OUTPUT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    create_openai_client,
    ensure_directories,
)

OUTPUT_DIR = METRICS_OUTPUT_DIR
ensure_directories()
client = create_openai_client()

# Metric passes verification if accuracy >= this threshold
ACCURACY_THRESHOLD = 0.9
# How many sample pairs to test each metric on
VERIFY_SAMPLE_SIZE = 10
# How many candidate metrics to propose per group
NUM_CANDIDATE_METRICS = 8
MAX_WORKERS_METRICS = 6
MAX_WORKERS_VERIFY = 5


def call_llm(prompt, system_msg="You are a precise AI assistant.", max_tokens=2048,
             temperature=0.1, max_retries=3):
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            print(f"    [retry {attempt + 1}] empty response")
            time.sleep(2)
        except Exception as exc:
            print(f"    [retry {attempt + 1}/{max_retries}] error: {exc}")
            time.sleep(2 ** attempt)
    return None


def extract_tasks_from_cluster(cluster_file, group_label):
    """Load cluster file and extract samples belonging to the given group."""
    with open(cluster_file, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    dim_id = cluster_data["dimension_id"]
    cluster_key = f"cluster_{dim_id}"

    samples = []
    for sample in cluster_data["samples"]:
        winner = sample.get("winner", "")
        if sample.get(cluster_key) == group_label and (
            "model_a" in winner or "model_b" in winner
        ):
            samples.append(sample)

    return samples, dim_id


# ─────────────────────────────────────────────────────────────────────
#  Stage 1: Propose metrics  (REWRITTEN — detailed & rigorous prompt)
# ─────────────────────────────────────────────────────────────────────

PROPOSE_SYSTEM = (
    "You are a senior AI evaluation researcher. You design rigorous, "
    "operationalizable scoring rubrics that are specific enough for two "
    "independent human raters to reach the same verdict on any given A/B pair."
)

PROPOSE_TEMPLATE = """You are designing a **detailed evaluation rubric** for comparing two AI assistant responses.

All the sample queries below share the same category:
  • Dimension: "{dim_id}"
  • Group: "{group_label}"

─── Sample Queries from this Group ───
{tasks_text}

─── Your Task ───

Propose exactly {n} evaluation metrics **tailored specifically to "{group_label}" queries**.

For EACH metric, provide:

1. **metric_name** — a short, unique snake_case identifier (e.g. `factual_accuracy`, `code_robustness`).

2. **metric_description** — a **thorough, self-contained paragraph (5-8 sentences)** that MUST cover ALL five aspects below:

   a) **Definition & Scope**: State precisely what quality dimension this metric captures and why it is important for "{group_label}" queries. Draw a clear boundary so the metric does not overlap with other metrics.

   b) **Positive Indicators**: List concrete, observable features of a response that performs well on this metric. Give specific examples of patterns, structures, content elements, or stylistic choices that a rater should look for.

   c) **Negative Indicators & Failure Modes**: List concrete, observable features that signal poor performance. Describe specific error types, omissions, structural problems, or misleading content that should lower the score.

   d) **Boundary & Edge Cases**: Describe ambiguous situations (e.g., both responses partially satisfy the criterion, or neither does) and provide explicit guidance on how to resolve them.

   e) **Tie-Breaking Guidance**: Specify under what conditions a rater should still prefer one response over the other versus declaring a tie. Include at least one concrete tie-breaking heuristic.

─── Constraints ───
- Do NOT use vague phrases like "overall quality", "better answer", "more helpful", or "more appropriate". Every sentence must reference **observable, verifiable** features.
- Each metric must be **clearly independent** — it must capture a distinctly different quality aspect from every other metric you propose.
- **Ground your metrics in the actual query patterns** shown above. For example, if queries involve code, reference code-specific criteria; if they involve creative writing, reference narrative or stylistic criteria.
- The description must be **self-contained**: a rater reading ONLY the metric_name and metric_description (with no other context) must be able to judge any A/B pair from this group.

─── Output Format (follow EXACTLY) ───

METRIC 1
metric_name: <snake_case_name>
metric_description: <detailed paragraph, 5-8 sentences, covering all five aspects a-e above>

METRIC 2
metric_name: <snake_case_name>
metric_description: <detailed paragraph, 5-8 sentences, covering all five aspects a-e above>

...continue until METRIC {n}...

Output NOTHING else — no preamble, no summary, no extra commentary."""


def propose_metrics(dim_id, group_label, sample_tasks):
    """Ask LLM to propose detailed candidate evaluation metrics for this group."""
    tasks_text = "\n".join(
        f"  [{i+1}] {task[:350].replace(chr(10), ' ')}"
        for i, task in enumerate(sample_tasks[:15])
    )

    prompt = PROPOSE_TEMPLATE.format(
        dim_id=dim_id,
        group_label=group_label,
        tasks_text=tasks_text,
        n=NUM_CANDIDATE_METRICS,
    )

    response = call_llm(
        prompt,
        system_msg=PROPOSE_SYSTEM,
        temperature=0.4,
        max_tokens=4096,
    )
    if not response:
        return []

    return _parse_proposed_metrics(response)


def _parse_proposed_metrics(text):
    """Parse proposed metrics from LLM output, supporting multi-line descriptions."""
    metrics = []
    current_name = None
    current_desc = None

    def _flush():
        nonlocal current_name, current_desc
        if current_name and current_desc:
            metrics.append({
                "metric_name": current_name,
                "metric_description": current_desc.strip(),
            })
        current_name = None
        current_desc = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue

        # "METRIC N" header → flush previous metric
        if re.match(r"^METRIC\s+\d+", stripped, re.IGNORECASE):
            _flush()
            continue

        # metric_name line
        name_match = re.match(
            r"(?:\d+[\.\)]\s*)?metric_name\s*:\s*(.+)", stripped, re.IGNORECASE
        )
        if name_match:
            _flush()
            current_name = name_match.group(1).strip().strip('"\'`*').lower()
            current_name = re.sub(r'[^a-z0-9_]', '_', current_name).strip('_')
            continue

        # metric_description first line
        desc_match = re.match(
            r"metric_description\s*:\s*(.+)", stripped, re.IGNORECASE
        )
        if desc_match:
            current_desc = desc_match.group(1).strip()
            continue

        # continuation lines of the description
        if current_name is not None and current_desc is not None:
            current_desc += " " + stripped

    _flush()
    return metrics


# ─────────────────────────────────────────────────────────────────────
#  Stage 2: Verify metrics  (UNCHANGED — original logic preserved)
# ─────────────────────────────────────────────────────────────────────

def verify_single_metric(metric, verify_samples):
    """
    Test one metric against sample A/B pairs.
    For each pair, ask LLM which response is better on THIS metric,
    then compare with ground truth.
    Returns accuracy (float).
    """
    def _verify_sample(sample):
        query = ""
        reply_a = ""
        reply_b = ""

        for turn in sample.get("conversation_a", []):
            if turn["role"] == "user":
                query = turn["content"][:800]
            elif turn["role"] == "assistant":
                reply_a = turn["content"][:1000]

        for turn in sample.get("conversation_b", []):
            if turn["role"] == "assistant":
                reply_b = turn["content"][:1000]

        if not query or not reply_a or not reply_b:
            return None

        # Determine ground truth
        winner = sample.get("winner", "")
        if "model_a" in winner:
            gt = "A"
        elif "model_b" in winner:
            gt = "B"
        else:
            return None

        prompt = f"""Evaluate which response is better based on this specific metric:

Metric: {metric['metric_name']}
Description: {metric['metric_description']}

[Query]: {query}

[Response A]: {reply_a}

[Response B]: {reply_b}

Based ONLY on the metric above, which response is better?
Output ONLY: A, B, or tie"""

        response = call_llm(
            prompt,
            system_msg="You are a metric-focused evaluator. Judge ONLY on the specified metric.",
            max_tokens=50,
            temperature=0.0,
        )

        if not response:
            return None

        pred = response.strip().upper()
        if pred.startswith("A"):
            pred = "A"
        elif pred.startswith("B"):
            pred = "B"
        else:
            pred = "tie"

        return pred, gt

    if not verify_samples:
        return 0, 0

    correct = 0
    total = 0

    max_workers = min(len(verify_samples), MAX_WORKERS_VERIFY)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_verify_sample, sample) for sample in verify_samples]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            pred, gt = result
            total += 1
            if pred == gt:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, total


# ─────────────────────────────────────────────────────────────────────
#  Orchestration  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────

def process_single_group(cluster_file, group_label):
    """
    Generate and verify metrics for one cluster group.

    Returns:
        dict with dimension_id, group_label, verified_metrics, etc.
        Or None if no valid metrics produced.
    """
    samples, dim_id = extract_tasks_from_cluster(cluster_file, group_label)

    if len(samples) < 3:
        print(f"    Too few samples ({len(samples)}), skipping")
        return None

    # Step 1: Extract task texts for metric proposal
    sample_tasks = []
    for s in samples[:30]:
        for turn in s.get("conversation_a", []):
            if turn["role"] == "user":
                sample_tasks.append(turn["content"])
                break

    if not sample_tasks:
        print("    No tasks extracted, skipping")
        return None

    # Step 2: Propose candidate metrics
    print(f"    Proposing metrics for {dim_id}/{group_label}...")
    candidates = propose_metrics(dim_id, group_label, sample_tasks)

    if not candidates:
        print("    No candidate metrics proposed")
        return None

    print(f"    Got {len(candidates)} candidate metrics:")
    for m in candidates:
        preview = m["metric_description"][:100] + "..." \
            if len(m["metric_description"]) > 100 else m["metric_description"]
        print(f"      - {m['metric_name']:30s}  {preview}")

    # Step 3: Verify each metric
    verify_samples = samples[:VERIFY_SAMPLE_SIZE]
    verified_metrics = []

    print(f"    Verifying on {len(verify_samples)} A/B pairs...")
    verification_results = [None] * len(candidates)
    max_workers = min(len(candidates), MAX_WORKERS_METRICS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(verify_single_metric, metric, verify_samples): idx
            for idx, metric in enumerate(candidates)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            verification_results[idx] = future.result()

    for metric, (accuracy, tested) in zip(candidates, verification_results):
        metric["accuracy"] = round(accuracy, 4)
        metric["tested_on"] = tested

        status = "PASS" if accuracy >= ACCURACY_THRESHOLD else "FAIL"
        print(
            f"      {status} {metric['metric_name']:30s}  "
            f"acc={accuracy:.0%} ({tested} samples)"
        )

        if accuracy >= ACCURACY_THRESHOLD:
            verified_metrics.append(metric)

    if not verified_metrics:
        # Keep top 2 even if below threshold, so we have something
        candidates.sort(key=lambda m: m.get("accuracy", 0), reverse=True)
        verified_metrics = candidates[:2]
        print(f"    No metrics passed threshold; keeping top {len(verified_metrics)} anyway")

    print(f"    Final: {len(verified_metrics)} metrics retained\n")

    result = {
        "dimension_id": dim_id,
        "group_label": group_label,
        "num_samples_in_group": len(samples),
        "num_candidate_metrics": len(candidates),
        "num_verified_metrics": len(verified_metrics),
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "all_candidates": candidates,
        "verified_metrics": verified_metrics,
    }

    return result


if __name__ == "__main__":
    import sys

    # Quick test: process the first group of the first dimension
    cluster_dir = os.path.join("results", "clusters")

    test_file = None
    for fname in os.listdir(cluster_dir):
        if fname.startswith("cluster_") and fname.endswith(".json"):
            test_file = os.path.join(cluster_dir, fname)
            break

    if not test_file:
        print("No cluster files found. Run clustering first.")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    first_group = next(iter(data["groups"].keys()))
    print(f"Testing with {test_file}, group={first_group}\n")

    result = process_single_group(test_file, first_group)

    if result:
        print(f"\nResult: {result['num_verified_metrics']} verified metrics")
        for m in result["verified_metrics"]:
            print(f"  - {m['metric_name']}: {m['metric_description'][:120]}... ({m['accuracy']:.0%})")
    else:
        print("\nNo result returned")
