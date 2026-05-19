"""
Cluster Chatbot Arena tasks along five LLM-generated dimensions.

This module keeps the original prompts and granularity-control logic:
initial classification, merging groups that are too small, and splitting
groups that are too large.
"""

import json
import math
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_from_disk

from config import (
    BATCH_SIZE,
    CLUSTER_OUTPUT_DIR,
    DATASET_PATH,
    MODEL_NAME,
    NUM_SAMPLES,
    SEED,
    create_openai_client,
    ensure_directories,
)


OUTPUT_DIR = CLUSTER_OUTPUT_DIR
MAX_WORKERS_BATCH = 8
ensure_directories()
client = create_openai_client()


def compute_group_bounds(total_n, num_labels):
    """
    Dynamically compute reasonable group-size bounds.

    Formula:
      ideal = N / K
      lower = max(5, ideal / 3, sqrt(N) / 2)
      upper = min(ideal * 3, N * 0.4)
    """
    ideal = total_n / num_labels
    lower = max(5, ideal / 3, math.sqrt(total_n) / 2)
    upper = min(ideal * 3, total_n * 0.4)

    if upper < lower:
        upper = max(lower * 2, total_n * 0.5)

    return int(math.ceil(lower)), int(math.floor(upper))


DIMENSIONS = [
    {
        "id": "task_type",
        "name": "Task Type",
        "description": "What type of task the user is asking the AI to perform",
        "labels": [
            "creative_writing",
            "knowledge_qa",
            "reasoning_math",
            "code_programming",
            "summarization_rewriting",
            "conversation_roleplay",
            "advice_planning",
            "extraction_analysis",
            "translation_language",
            "other",
        ],
        "prompt_template": """Classify each user query into ONE of these task types:
- creative_writing: Creative writing, stories, poems, scripts, fiction
- knowledge_qa: Factual questions, knowledge queries, definitions, explanations
- reasoning_math: Logical reasoning, math problems, puzzles, analytical thinking
- code_programming: Code generation, debugging, programming help, algorithms
- summarization_rewriting: Summarize, paraphrase, rewrite, edit text
- conversation_roleplay: Casual chat, role-play, persona simulation, social
- advice_planning: Advice, planning, decision-making, how-to guides, recommendations
- extraction_analysis: Data extraction, comparison, analysis, evaluation
- translation_language: Translation, grammar correction, language learning
- other: Doesn't clearly fit above categories

For each query, output ONLY the label, one per line:
[1] label
[2] label
...

Queries:
{queries}""",
    },
    {
        "id": "difficulty",
        "name": "Difficulty Level",
        "description": "How difficult/complex the user query is for an AI to answer well",
        "labels": [
            "easy",
            "medium",
            "hard",
            "very_hard",
        ],
        "prompt_template": """Rate the difficulty of each query for an AI assistant to answer WELL:
- easy: Simple greetings, basic facts, straightforward short requests
- medium: Requires moderate knowledge, some organization, standard tasks
- hard: Complex reasoning, professional knowledge, detailed creation, multi-constraints
- very_hard: Multi-step expert reasoning, cutting-edge topics, highly specialized, ambiguous

For each query, output ONLY the label, one per line:
[1] label
[2] label
...

Queries:
{queries}""",
    },
    {
        "id": "domain",
        "name": "Domain/Topic",
        "description": "The subject domain or topic area of the query",
        "labels": [
            "science_tech",
            "humanities_social",
            "daily_life",
            "business_finance",
            "education_academic",
            "entertainment_culture",
            "law_politics",
            "creative_arts",
            "general",
        ],
        "prompt_template": """Classify each user query into ONE primary domain/topic:
- science_tech: Science, technology, engineering, computer science, math, medicine
- humanities_social: History, philosophy, sociology, psychology, linguistics, literature
- daily_life: Daily life, food, travel, health, fitness, relationships, household
- business_finance: Business, finance, economics, marketing, startups, careers
- education_academic: Education, studying, exams, academic writing, research methods
- entertainment_culture: Games, movies, music, sports, pop culture, hobbies
- law_politics: Law, politics, policy, government, ethics, regulations
- creative_arts: Art, design, photography, creative expression, aesthetics
- general: General purpose, meta-questions about AI, no clear single domain

For each query, output ONLY the label, one per line:
[1] label
[2] label
...

Queries:
{queries}""",
    },
    {
        "id": "response_format",
        "name": "Expected Response Format",
        "description": "What format/structure the user implicitly or explicitly expects",
        "labels": [
            "short_answer",
            "explanation",
            "list_steps",
            "long_form",
            "code_structured",
            "open_ended",
        ],
        "prompt_template": """Classify what response FORMAT each user query expects:
- short_answer: Brief answer - a few words, a number, yes/no, one sentence
- explanation: Explanatory paragraphs - teaching, clarifying, defining concepts
- list_steps: Bullet points, numbered lists, step-by-step instructions
- long_form: Long essays, articles, reports, detailed stories (multiple paragraphs)
- code_structured: Code snippets, JSON, tables, structured data output
- open_ended: Open-ended creative response, no specific format implied

For each query, output ONLY the label, one per line:
[1] label
[2] label
...

Queries:
{queries}""",
    },
    {
        "id": "capability",
        "name": "Core Capability Required",
        "description": "Which AI capability is MOST critically tested by this query",
        "labels": [
            "factual_accuracy",
            "logical_reasoning",
            "creativity_imagination",
            "instruction_following",
            "communication_clarity",
            "domain_expertise",
            "empathy_tone",
            "multilingual",
        ],
        "prompt_template": """For each query, identify the SINGLE most important AI capability being tested:
- factual_accuracy: Getting facts right, precise knowledge, avoiding hallucination
- logical_reasoning: Step-by-step logic, math, deduction, causal analysis
- creativity_imagination: Novel ideas, creative writing, brainstorming, originality
- instruction_following: Following complex/specific instructions, constraints, formatting rules
- communication_clarity: Clear explanation, good structure, readability, appropriate detail level
- domain_expertise: Deep professional/specialized knowledge (law, medicine, finance, etc.)
- empathy_tone: Emotional intelligence, appropriate tone, sensitivity, social awareness
- multilingual: Translation quality, cross-lingual understanding, language-specific nuance

For each query, output ONLY the label, one per line:
[1] label
[2] label
...

Queries:
{queries}""",
    },
]


def extract_task(conversation_a):
    """Extract the first user query from conversation_a."""
    for turn in conversation_a:
        if turn["role"] == "user":
            return turn["content"]
    return ""


def _is_non_tie_winner(winner):
    return "model_a" in winner or "model_b" in winner


def format_batch_queries(tasks, start_idx):
    """Format a batch of tasks as numbered lines."""
    lines = []
    for j, task in enumerate(tasks):
        text = task[:500] + "..." if len(task) > 500 else task
        text = text.replace("\n", " ").strip()
        lines.append(f"[{start_idx + j + 1}] {text}")
    return "\n".join(lines)


def parse_batch_labels(response_text, batch_size, valid_labels):
    """Parse model output into a list of valid labels."""
    labels = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        cleaned = re.sub(r"^\[?\d+\]?[\.\)\:\-\s]*", "", line).strip().lower()
        cleaned = cleaned.strip('"\'`*')

        matched = None
        for label in valid_labels:
            if label.lower() == cleaned:
                matched = label
                break
        if not matched:
            for label in valid_labels:
                if label.lower() in cleaned:
                    matched = label
                    break
        if not matched:
            best_score = 0
            for label in valid_labels:
                score = sum(1 for word in label.split("_") if word in cleaned)
                if score > best_score:
                    best_score = score
                    matched = label
        if not matched:
            matched = valid_labels[-1]

        labels.append(matched)

    while len(labels) < batch_size:
        labels.append(valid_labels[-1])
    return labels[:batch_size]


def call_llm_batch(prompt, max_retries=3):
    """Call the LLM for classification-style prompts."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise classifier. Output ONLY labels, one per line, no explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            print(f"  [retry {attempt + 1}] empty response")
            time.sleep(2)
        except Exception as exc:
            print(f"  [retry {attempt + 1}/{max_retries}] error: {exc}")
            time.sleep(2**attempt)
    return None


def split_large_group(samples_in_group, original_label, dim_config, upper_bound):
    """
    Ask the LLM to split a too-large group into sub-labels.

    Returns a list of sub-labels aligned with samples_in_group, or None.
    """
    n = len(samples_in_group)
    target_sub_groups = max(2, math.ceil(n / upper_bound) + 1)

    tasks = [sample["task"] for sample in samples_in_group]
    sample_tasks = tasks[:30]
    tasks_text = "\n".join(
        f"[{i + 1}] {task[:300].replace(chr(10), ' ')}"
        for i, task in enumerate(sample_tasks)
    )

    discover_prompt = f"""These {len(sample_tasks)} queries all belong to the category "{original_label}".
This group is too large. Please split it into {target_sub_groups}-{target_sub_groups+2} meaningful sub-categories.

Sample queries:
{tasks_text}

Output ONLY the sub-category names (short, snake_case), one per line. No numbering, no explanation.
Example format:
sub_category_one
sub_category_two
sub_category_three"""

    response = call_llm_batch(discover_prompt)
    if not response:
        return None

    sub_labels = [
        line.strip().lower().strip('"\'`*-. ')
        for line in response.strip().split("\n")
        if line.strip()
    ]
    sub_labels = [label for label in sub_labels if label and len(label) < 50]

    if len(sub_labels) < 2:
        return None

    classify_template = f"""Classify each query into ONE of these sub-categories of "{original_label}":
{chr(10).join(f'- {sub_label}' for sub_label in sub_labels)}

For each query, output ONLY the sub-label, one per line:
[1] label
[2] label
...

Queries:
{{queries}}"""

    all_sub_labels = []
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_tasks = tasks[batch_start:batch_end]
        queries_text = format_batch_queries(batch_tasks, batch_start)
        prompt = classify_template.format(queries=queries_text)
        response = call_llm_batch(prompt)
        if response:
            batch_labels = parse_batch_labels(response, len(batch_tasks), sub_labels)
        else:
            batch_labels = [sub_labels[0]] * len(batch_tasks)
        all_sub_labels.extend(batch_labels)

    return all_sub_labels


def merge_small_groups(labels_list, valid_labels, lower_bound):
    """
    Merge too-small groups into larger groups in the same dimension.
    """
    counts = Counter(labels_list)
    small_groups = [label for label, count in counts.items() if count < lower_bound]
    large_groups = [label for label, count in counts.items() if count >= lower_bound]

    if not small_groups or not large_groups:
        return labels_list

    merge_prompt = f"""These are category labels. Some groups are too small and need to be merged into larger groups.

Large groups (keep these): {', '.join(large_groups)}
Small groups (merge these): {', '.join(small_groups)}

For each small group, which large group is it most similar to?
Output format (one per line):
small_label -> large_label

Only output the mappings, nothing else."""

    response = call_llm_batch(merge_prompt)

    merge_map = {}
    if response:
        for line in response.strip().split("\n"):
            line = line.strip()
            if "->" in line:
                parts = line.split("->")
                src = parts[0].strip().lower().strip('"\'`* ')
                dst = parts[1].strip().lower().strip('"\'`* ')
                if dst in [label.lower() for label in large_groups]:
                    merge_map[src] = dst

    largest = max(large_groups, key=lambda label: counts[label])
    for small_group in small_groups:
        if small_group.lower() not in merge_map:
            merge_map[small_group.lower()] = largest.lower()

    new_labels = []
    for label in labels_list:
        if label.lower() in merge_map:
            target_lower = merge_map[label.lower()]
            target = next(
                (large_group for large_group in large_groups if large_group.lower() == target_lower),
                label,
            )
            new_labels.append(target)
        else:
            new_labels.append(label)

    return new_labels


def cluster_one_dimension(dim_config, samples):
    """Cluster samples along one dimension with granularity control."""
    dim_name = dim_config["name"]
    labels_def = dim_config["labels"]
    template = dim_config["prompt_template"]

    total_n = len(samples)
    lower_bound, upper_bound = compute_group_bounds(total_n, len(labels_def))

    print(f"\n{'=' * 60}")
    print(f"  Dimension: {dim_name}")
    print(f"  Available labels: {len(labels_def)}")
    print(f"  Granularity bounds: [{lower_bound}, {upper_bound}] samples per group")
    print(f"  (N={total_n}, K={len(labels_def)}, ideal={total_n / len(labels_def):.0f})")
    print(f"{'=' * 60}")

    print("\n  [Round 1] Initial classification...")
    batch_jobs = []
    for batch_start in range(0, total_n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_n)
        batch_tasks = [sample["task"] for sample in samples[batch_start:batch_end]]
        queries_text = format_batch_queries(batch_tasks, batch_start)
        prompt = template.format(queries=queries_text)

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = math.ceil(total_n / BATCH_SIZE)
        batch_jobs.append((batch_num, total_batches, batch_tasks, prompt))

    def _classify_batch(batch_num, total_batches, batch_tasks, prompt):
        response = call_llm_batch(prompt)
        if response:
            batch_labels = parse_batch_labels(response, len(batch_tasks), labels_def)
            status = "ok"
        else:
            batch_labels = [labels_def[-1]] * len(batch_tasks)
            status = "fallback"
        return batch_num, total_batches, batch_labels, status

    batch_results = [None] * len(batch_jobs)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_BATCH) as executor:
        future_to_index = {
            executor.submit(_classify_batch, *job): idx
            for idx, job in enumerate(batch_jobs)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            batch_num, total_batches, batch_labels, status = future.result()
            batch_results[idx] = batch_labels
            print(f"    Batch {batch_num}/{total_batches} {status}")

    all_labels = []
    for batch_labels in batch_results:
        all_labels.extend(batch_labels)

    counts = Counter(all_labels)
    print("\n  Initial distribution:")
    for label, count in counts.most_common():
        bar = "#" * (count * 40 // total_n)
        flag = ""
        if count > upper_bound:
            flag = " too_large"
        elif count < lower_bound:
            flag = " too_small"
        print(f"    {label:30s}: {count:4d} ({count / total_n * 100:5.1f}%) {bar}{flag}")

    small_groups = [label for label, count in counts.items() if count < lower_bound]
    if small_groups:
        print(f"\n  [Round 2] Merge too-small groups: {small_groups}")
        all_labels = merge_small_groups(all_labels, labels_def, lower_bound)
        counts = Counter(all_labels)
        print("  Distribution after merge:")
        for label, count in counts.most_common():
            bar = "#" * (count * 40 // total_n)
            print(f"    {label:30s}: {count:4d} ({count / total_n * 100:5.1f}%) {bar}")

    large_groups = [label for label, count in counts.items() if count > upper_bound]
    if large_groups:
        print(f"\n  [Round 3] Split too-large groups: {large_groups}")

        for large_label in large_groups:
            large_indices = [i for i, label in enumerate(all_labels) if label == large_label]
            large_samples = [samples[i] for i in large_indices]
            large_count = len(large_indices)

            print(f"    Splitting '{large_label}' ({large_count} samples)...")

            sub_labels = split_large_group(large_samples, large_label, dim_config, upper_bound)

            if sub_labels and len(sub_labels) == len(large_indices):
                for j, original_idx in enumerate(large_indices):
                    all_labels[original_idx] = f"{large_label}/{sub_labels[j]}"
                sub_counts = Counter(sub_labels)
                print(f"      Split into {len(sub_counts)} subgroups: {dict(sub_counts.most_common())}")
            else:
                print("      Split failed; keeping original label")

        counts = Counter(all_labels)
        print("\n  Final distribution:")
        for label, count in counts.most_common():
            bar = "#" * (count * 40 // total_n)
            print(f"    {label:30s}: {count:4d} ({count / total_n * 100:5.1f}%) {bar}")

    return all_labels


def main():
    print("Testing API connection...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=10,
        )
        print(f"API ok: {response.choices[0].message.content}\n")
    except Exception as exc:
        print(f"API connection failed: {exc}")
        return

    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    data = dataset["train"]
    print(f"Total rows: {len(data)}")

    random.seed(SEED)
    eligible_indices = [
        idx for idx in range(len(data))
        if _is_non_tie_winner(data[idx]["winner"])
    ]
    indices = random.sample(eligible_indices, min(NUM_SAMPLES, len(eligible_indices)))

    samples = []
    for idx in indices:
        row = data[idx]
        if not _is_non_tie_winner(row["winner"]):
            continue
        task = extract_task(row["conversation_a"])
        samples.append(
            {
                "index": idx,
                "question_id": row["question_id"],
                "task": task,
                "conversation_a": row["conversation_a"],
                "conversation_b": row["conversation_b"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "winner": row["winner"],
            }
        )

    print(f"Sampled {len(samples)} rows")
    print(f"Task example: {samples[0]['task'][:120]}...\n")

    print("=" * 60)
    print("  Granularity-control formula")
    print("  lower = max(5, N/(K*3), sqrt(N)/2)")
    print("  upper = min(N/K*3, N*0.4)")
    for dim in DIMENSIONS:
        lower, upper = compute_group_bounds(len(samples), len(dim["labels"]))
        print(f"  {dim['id']:20s}: K={len(dim['labels']):2d}, bounds=[{lower:3d}, {upper:3d}]")
    print("=" * 60)

    start_time = time.time()

    for dim_config in DIMENSIONS:
        dim_id = dim_config["id"]
        dim_start = time.time()

        labels = cluster_one_dimension(dim_config, samples)
        dim_elapsed = time.time() - dim_start

        counts = Counter(labels)
        groups = {}
        output_samples = []

        for i, sample in enumerate(samples):
            label = labels[i]
            output_samples.append(
                {
                    "index": sample["index"],
                    "question_id": sample["question_id"],
                    "task": sample["task"],
                    "conversation_a": sample["conversation_a"],
                    "conversation_b": sample["conversation_b"],
                    "model_a": sample["model_a"],
                    "model_b": sample["model_b"],
                    "winner": sample["winner"],
                    f"cluster_{dim_id}": label,
                }
            )
            groups.setdefault(label, []).append(sample["question_id"])

        lower_bound, upper_bound = compute_group_bounds(len(samples), len(dim_config["labels"]))

        output = {
            "dimension_id": dim_id,
            "dimension_name": dim_config["name"],
            "description": dim_config["description"],
            "available_labels": dim_config["labels"],
            "granularity": {
                "formula": "lower=max(5, N/(K*3), sqrt(N)/2), upper=min(N/K*3, N*0.4)",
                "N": len(samples),
                "K": len(dim_config["labels"]),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            "label_distribution": dict(counts.most_common()),
            "num_groups": len(counts),
            "num_samples": len(samples),
            "groups": {
                label: {
                    "count": len(question_ids),
                    "within_bounds": lower_bound <= len(question_ids) <= upper_bound,
                    "question_ids": question_ids,
                }
                for label, question_ids in groups.items()
            },
            "samples": output_samples,
            "elapsed_seconds": round(dim_elapsed, 1),
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"cluster_{dim_id}.json")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
        print(f"\n  Saved {output_file} (elapsed {dim_elapsed:.1f}s)")

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Done. Total elapsed: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print("\nGenerated files:")
    for dim in DIMENSIONS:
        file_path = os.path.join(OUTPUT_DIR, f"cluster_{dim['id']}.json")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024 / 1024
            with open(file_path, "r", encoding="utf-8") as file:
                meta = json.load(file)
            bounds = meta["granularity"]
            in_bounds = sum(1 for group in meta["groups"].values() if group["within_bounds"])
            total_groups = len(meta["groups"])
            print(f"\n  {file_path} ({file_size:.1f} MB)")
            print(f"     {total_groups} groups; {in_bounds}/{total_groups} within bounds")
            print(f"     Bounds: [{bounds['lower_bound']}, {bounds['upper_bound']}]")
            for label, count in meta["label_distribution"].items():
                flag = "ok" if bounds["lower_bound"] <= count <= bounds["upper_bound"] else "check"
                print(f"     {flag} {label}: {count}")


if __name__ == "__main__":
    main()
