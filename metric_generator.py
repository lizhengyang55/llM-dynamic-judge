# generate_metrics.py
"""
读取聚类文件的指定组，分析回答模式，生成具体的评判 metric。
每个 metric 要求能正确判断该组 >= 60% 的 winner。
输出: (分组依据, 具体标签, metric, 可正确判断的样本列表)
"""

import os
import json
import time
import math
import random
import re
from collections import Counter
from openai import OpenAI
from pathlib import Path

# ===================== 配置区 =====================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

api_key = Path(r"E:\homework\ai\apikey.txt").read_text(encoding="utf-8").strip()
BASE_URL = "https://yeysai.com/v1"
MODEL_NAME = "gpt-4o-mini"

CLUSTER_DIR = "results/clusters"
OUTPUT_DIR = "results/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成 metric 时展示的样本数（太多会超 token）
EXAMPLES_FOR_GENERATION = 15
# 验证时每批多少条
VERIFY_BATCH_SIZE = 5
# metric 最少正确率
MIN_ACCURACY = 0.60
# 每组期望生成多少个 metric
TARGET_METRICS_PER_GROUP = 5

SEED = 42
random.seed(SEED)
# ==================================================

client = OpenAI(api_key=api_key, base_url=BASE_URL)


def call_llm(prompt, system_msg="You are a precise AI evaluation expert.",
             max_tokens=4096, max_retries=3):
    """调用大模型"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 if attempt == 0 else 0.5,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            print(f"    [重试 {attempt+1}] 空响应")
            time.sleep(2)
        except Exception as e:
            print(f"    [重试 {attempt+1}/{max_retries}] {e}")
            time.sleep(2 ** attempt)
    return None


def format_sample_for_display(sample, idx, show_winner=True):
    """格式化单条样本用于展示给 LLM"""

    def extract_assistant_reply(conv):
        parts = []
        for turn in conv:
            if turn["role"] == "assistant":
                content = turn["content"]
                if len(content) > 800:
                    content = content[:800] + "...[truncated]"
                parts.append(content)
        return "\n".join(parts) if parts else "[empty]"

    def extract_user_query(conv):
        for turn in conv:
            if turn["role"] == "user":
                content = turn["content"]
                if len(content) > 500:
                    content = content[:500] + "...[truncated]"
                return content
        return "[empty]"

    query = extract_user_query(sample["conversation_a"])
    reply_a = extract_assistant_reply(sample["conversation_a"])
    reply_b = extract_assistant_reply(sample["conversation_b"])

    text = f"""--- Example {idx} ---
[User Query]: {query}

[Assistant A ({sample['model_a']})]:
{reply_a}

[Assistant B ({sample['model_b']})]:
{reply_b}
"""
    if show_winner:
        winner = sample["winner"]
        if "model_a" in winner:
            text += f"\n[Winner]: Assistant A\n"
        elif "model_b" in winner:
            text += f"\n[Winner]: Assistant B\n"
        else:
            text += f"\n[Winner]: Tie\n"

    return text


def generate_metrics_for_group(dimension_id, dimension_name, group_label,
                               group_samples):
    """
    分析一组样本，生成多个具体的评判 metric。
    
    关键：metric 不能是"哪个更好"这种废话，
    而要是具体的、可操作的评判标准。
    """
    n = len(group_samples)

    # 分析 winner 分布
    winner_counts = Counter()
    for s in group_samples:
        w = s["winner"]
        if "model_a" in w:
            winner_counts["A"] += 1
        elif "model_b" in w:
            winner_counts["B"] += 1
        else:
            winner_counts["tie"] += 1

    print(f"    Winner分布: A={winner_counts.get('A',0)}, "
          f"B={winner_counts.get('B',0)}, tie={winner_counts.get('tie',0)}")

    # 选取展示样本（确保 winner 多样性）
    display_samples = random.sample(
        group_samples, min(EXAMPLES_FOR_GENERATION, n))

    examples_text = ""
    for i, s in enumerate(display_samples):
        examples_text += format_sample_for_display(s, i + 1, show_winner=True)
        examples_text += "\n"

    # ===== 第一步：让 LLM 分析模式并生成 metric =====
    gen_prompt = f"""You are an expert in AI evaluation. I will show you {len(display_samples)} examples from a specific category of user queries.

Category context:
- Clustering dimension: {dimension_name}
- Group label: {group_label}

Each example shows: User Query, Assistant A's response, Assistant B's response, and which one won (chosen by human voters).

Your task: Analyze the PATTERNS in what makes a response win in this category, then propose {TARGET_METRICS_PER_GROUP + 3} specific, detailed evaluation METRICS.

CRITICAL REQUIREMENTS for each metric:
1. Be SPECIFIC and DETAILED — not vague like "better quality" or "more helpful"
2. Must be a concrete, observable criterion that can be checked by reading the responses
3. Each metric should describe a SPECIFIC aspect with clear scoring guidance
4. The metric should be phrased as a COMPARATIVE judgment rule (A vs B)
5. Each metric description must be at least 3-4 sentences long with concrete examples of what to look for

BAD metric example (too vague): "Which response is more accurate"
GOOD metric example: "Factual Completeness and Precision: Check whether the response covers ALL key aspects of the question without omitting important sub-topics. Count the number of distinct factual claims made. Verify that specific numbers, dates, or technical terms are used correctly rather than approximated. A response that addresses 3 out of 4 sub-questions with precise details should score higher than one that vaguely touches all 4."

{examples_text}

Now output EXACTLY in this JSON format (no other text):
```json
[
  {{
    "metric_id": "metric_1",
    "metric_name": "Short Name (3-6 words)",
    "metric_description": "Detailed 3-5 sentence description of what exactly to evaluate, what constitutes doing well vs poorly, with concrete observable criteria...",
    "pattern_observed": "1-2 sentences explaining what pattern in the examples led you to propose this metric"
  }},
  ...
]
```"""

    print(f"    生成 metrics...")
    response = call_llm(gen_prompt, max_tokens=4096)

    if not response:
        print(f"    ❌ 生成失败")
        return []

    # 解析 JSON
    metrics = parse_metrics_json(response)
    if not metrics:
        print(f"    ❌ JSON 解析失败")
        print(f"    原始响应前500字: {response[:500]}")
        return []

    print(f"    生成了 {len(metrics)} 个候选 metric")
    return metrics


def parse_metrics_json(response_text):
    """从 LLM 响应中提取 metric JSON"""
    # 尝试提取 ```json ... ``` 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接找 [ ... ]
        bracket_match = re.search(r'\[[\s\S]*\]', response_text)
        if bracket_match:
            json_str = bracket_match.group(0)
        else:
            return None

    try:
        metrics = json.loads(json_str)
        if isinstance(metrics, list):
            return metrics
    except json.JSONDecodeError:
        # 尝试修复常见问题
        try:
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            metrics = json.loads(json_str)
            if isinstance(metrics, list):
                return metrics
        except:
            pass
    return None


def verify_metric_on_samples(metric, group_samples):
    """
    用一个 metric 逐批验证所有样本，返回正确判断的样本列表。
    让 LLM 按照该 metric 判断每对回答的 winner。
    """
    metric_name = metric["metric_name"]
    metric_desc = metric["metric_description"]

    correct_ids = []
    incorrect_ids = []
    all_predictions = []

    # 分批验证
    for batch_start in range(0, len(group_samples), VERIFY_BATCH_SIZE):
        batch_end = min(batch_start + VERIFY_BATCH_SIZE, len(group_samples))
        batch = group_samples[batch_start:batch_end]

        # 构造验证 prompt
        samples_text = ""
        for j, s in enumerate(batch):
            samples_text += format_sample_for_display(
                s, batch_start + j + 1, show_winner=False)
            samples_text += "\n"

        verify_prompt = f"""You are evaluating AI responses using this SPECIFIC metric:

METRIC: {metric_name}
DESCRIPTION: {metric_desc}

Based ONLY on this metric, judge which assistant is better for each example below.

{samples_text}

For each example, output your judgment in this exact format (one per line):
[1] A or B or tie
[2] A or B or tie
...

Output ONLY the judgments, no explanations."""

        response = call_llm(verify_prompt, max_tokens=512)

        if not response:
            continue

        # 解析判断
        judgments = parse_verify_response(response, len(batch))

        for j, (judgment, sample) in enumerate(zip(judgments, batch)):
            gt_winner = sample["winner"]
            if "model_a" in gt_winner:
                gt = "A"
            elif "model_b" in gt_winner:
                gt = "B"
            else:
                gt = "tie"

            is_correct = (judgment == gt)

            pred_record = {
                "question_id": sample["question_id"],
                "ground_truth": gt,
                "prediction": judgment,
                "correct": is_correct,
            }
            all_predictions.append(pred_record)

            if is_correct:
                correct_ids.append(sample["question_id"])
            else:
                incorrect_ids.append(sample["question_id"])

    return correct_ids, incorrect_ids, all_predictions


def parse_verify_response(response_text, expected_count):
    """解析验证响应"""
    judgments = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^\[?\d+\]?[\.\)\:\-\s]*', '', line).strip().upper()

        if cleaned.startswith("A"):
            judgments.append("A")
        elif cleaned.startswith("B"):
            judgments.append("B")
        elif cleaned.startswith("TIE") or cleaned.startswith("T"):
            judgments.append("tie")
        elif "A" in cleaned and "B" not in cleaned:
            judgments.append("A")
        elif "B" in cleaned and "A" not in cleaned:
            judgments.append("B")
        else:
            judgments.append("tie")

    while len(judgments) < expected_count:
        judgments.append("tie")
    return judgments[:expected_count]


def process_single_group(cluster_file, group_label):
    """处理单个聚类文件的单个组"""

    print(f"\n加载: {cluster_file}")
    with open(cluster_file, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    dimension_id = cluster_data["dimension_id"]
    dimension_name = cluster_data["dimension_name"]

    # 提取该组样本
    label_key = f"cluster_{dimension_id}"
    group_samples = [s for s in cluster_data["samples"]
                     if s.get(label_key) == group_label]

    if not group_samples:
        print(f"  ❌ 组 '{group_label}' 无样本")
        return None

    n = len(group_samples)
    print(f"  维度: {dimension_name}")
    print(f"  标签: {group_label}")
    print(f"  样本数: {n}")

    # 过滤掉 tie 样本（tie 很难判断，影响 metric 质量）
    non_tie_samples = [s for s in group_samples
                       if "tie" not in s["winner"]]
    tie_samples = [s for s in group_samples if "tie" in s["winner"]]
    print(f"  非 tie 样本: {len(non_tie_samples)}, tie: {len(tie_samples)}")

    if len(non_tie_samples) < 6:
        print(f"  ⚠️ 非 tie 样本太少，跳过")
        return None

    # ===== 生成 metrics =====
    candidate_metrics = generate_metrics_for_group(
        dimension_id, dimension_name, group_label, non_tie_samples)

    if not candidate_metrics:
        return None

    # ===== 验证每个 metric =====
    verified_metrics = []

    for m_idx, metric in enumerate(candidate_metrics):
        metric_name = metric.get("metric_name", f"metric_{m_idx}")
        print(f"\n    验证 metric {m_idx+1}/{len(candidate_metrics)}: "
              f"{metric_name}")

        correct_ids, incorrect_ids, predictions = verify_metric_on_samples(
            metric, non_tie_samples)

        total_verified = len(correct_ids) + len(incorrect_ids)
        if total_verified == 0:
            print(f"      ❌ 无有效验证结果")
            continue

        accuracy = len(correct_ids) / total_verified
        print(f"      准确率: {len(correct_ids)}/{total_verified} "
              f"= {accuracy:.1%}")

        if accuracy >= MIN_ACCURACY:
            print(f"      ✅ 通过! (>= {MIN_ACCURACY:.0%})")
            verified_metrics.append({
                "metric_id": metric.get("metric_id", f"metric_{m_idx+1}"),
                "metric_name": metric_name,
                "metric_description": metric.get("metric_description", ""),
                "pattern_observed": metric.get("pattern_observed", ""),
                "accuracy": round(accuracy, 4),
                "correct_count": len(correct_ids),
                "total_verified": total_verified,
                "correct_question_ids": correct_ids,
                "incorrect_question_ids": incorrect_ids,
                "predictions": predictions,
            })
        else:
            print(f"      ❌ 未通过 ({accuracy:.1%} < {MIN_ACCURACY:.0%})")

    print(f"\n  📊 结果: {len(verified_metrics)}/{len(candidate_metrics)} "
          f"个 metric 通过验证")

    # 构建输出
    result = {
        "dimension_id": dimension_id,
        "dimension_name": dimension_name,
        "group_label": group_label,
        "num_samples_total": n,
        "num_samples_non_tie": len(non_tie_samples),
        "num_samples_tie": len(tie_samples),
        "num_candidate_metrics": len(candidate_metrics),
        "num_verified_metrics": len(verified_metrics),
        "min_accuracy_threshold": MIN_ACCURACY,
        "verified_metrics": sorted(verified_metrics,
                                    key=lambda x: x["accuracy"],
                                    reverse=True),
    }

    return result


def process_all_groups_in_file(cluster_file):
    """处理一个聚类文件中的所有组"""

    with open(cluster_file, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    dimension_id = cluster_data["dimension_id"]
    groups = cluster_data["groups"]

    print(f"\n{'='*60}")
    print(f"  处理维度: {cluster_data['dimension_name']}")
    print(f"  共 {len(groups)} 个组")
    print(f"{'='*60}")

    all_results = []

    for group_label, group_info in groups.items():
        count = group_info["count"]
        print(f"\n{'─'*50}")
        print(f"  组: {group_label} ({count} 条)")

        if count < 6:
            print(f"  ⚠️ 样本太少，跳过")
            continue

        result = process_single_group(cluster_file, group_label)
        if result:
            all_results.append(result)

            # 每组处理完就保存（防崩溃丢失）
            interim_file = os.path.join(
                OUTPUT_DIR,
                f"metrics_{dimension_id}_{group_label.replace('/', '_')}.json"
            )
            with open(interim_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  💾 已保存: {interim_file}")

    return all_results


def main():
    """
    使用方式:
      1. 处理单个文件的单个组:
         python generate_metrics.py --file cluster_task_type.json --group knowledge_qa

      2. 处理单个文件的所有组:
         python generate_metrics.py --file cluster_task_type.json

      3. 处理所有文件的所有组:
         python generate_metrics.py --all
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="聚类文件名 (在 results/clusters/ 下)")
    parser.add_argument("--group", type=str, default=None,
                        help="指定组标签")
    parser.add_argument("--all", action="store_true",
                        help="处理所有聚类文件的所有组")
    args = parser.parse_args()

    # 测试API
    print("测试API连接...")
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=10,
        )
        print(f"✅ API正常: {resp.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return

    start_time = time.time()

    if args.all:
        # 处理所有文件
        cluster_files = sorted(Path(CLUSTER_DIR).glob("cluster_*.json"))
        print(f"找到 {len(cluster_files)} 个聚类文件")

        grand_results = []
        for cf in cluster_files:
            results = process_all_groups_in_file(str(cf))
            grand_results.extend(results)

        # 保存汇总
        summary_file = os.path.join(OUTPUT_DIR, "all_metrics_summary.json")
        summary = {
            "total_groups_processed": len(grand_results),
            "total_metrics_generated": sum(
                r["num_verified_metrics"] for r in grand_results),
            "results_by_group": [
                {
                    "dimension": r["dimension_id"],
                    "group": r["group_label"],
                    "num_metrics": r["num_verified_metrics"],
                    "best_accuracy": (r["verified_metrics"][0]["accuracy"]
                                      if r["verified_metrics"] else 0),
                    "metrics_summary": [
                        {
                            "name": m["metric_name"],
                            "accuracy": m["accuracy"],
                            "correct_count": m["correct_count"],
                        }
                        for m in r["verified_metrics"]
                    ]
                }
                for r in grand_results
            ]
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n📋 汇总已保存: {summary_file}")

    elif args.file:
        cluster_file = os.path.join(CLUSTER_DIR, args.file)
        if not os.path.exists(cluster_file):
            print(f"❌ 文件不存在: {cluster_file}")
            return

        if args.group:
            result = process_single_group(cluster_file, args.group)
            if result:
                out_file = os.path.join(
                    OUTPUT_DIR,
                    f"metrics_{result['dimension_id']}_{args.group.replace('/', '_')}.json"
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n💾 已保存: {out_file}")
        else:
            process_all_groups_in_file(cluster_file)

    else:
        # 默认：处理所有
        print("未指定参数，默认处理所有聚类文件\n")
        print("用法示例:")
        print("  python generate_metrics.py --all")
        print("  python generate_metrics.py --file cluster_task_type.json")
        print("  python generate_metrics.py --file cluster_task_type.json --group knowledge_qa")
        print("\n自动执行 --all ...\n")

        cluster_files = sorted(Path(CLUSTER_DIR).glob("cluster_*.json"))
        if not cluster_files:
            print(f"❌ {CLUSTER_DIR}/ 下没有聚类文件，请先运行 cluster_by_dimensions.py")
            return

        print(f"找到 {len(cluster_files)} 个聚类文件")

        grand_results = []
        for cf in cluster_files:
            results = process_all_groups_in_file(str(cf))
            grand_results.extend(results)

        # 保存汇总
        summary_file = os.path.join(OUTPUT_DIR, "all_metrics_summary.json")

        total_metrics = sum(r["num_verified_metrics"] for r in grand_results)
        avg_accuracy = 0
        if total_metrics > 0:
            all_accs = [m["accuracy"]
                        for r in grand_results
                        for m in r["verified_metrics"]]
            avg_accuracy = sum(all_accs) / len(all_accs)

        summary = {
            "config": {
                "model": MODEL_NAME,
                "min_accuracy_threshold": MIN_ACCURACY,
                "target_metrics_per_group": TARGET_METRICS_PER_GROUP,
                "examples_for_generation": EXAMPLES_FOR_GENERATION,
            },
            "overview": {
                "total_groups_processed": len(grand_results),
                "total_metrics_generated": total_metrics,
                "average_metric_accuracy": round(avg_accuracy, 4),
            },
            "results_by_group": [
                {
                    "dimension_id": r["dimension_id"],
                    "dimension_name": r["dimension_name"],
                    "group_label": r["group_label"],
                    "num_samples": r["num_samples_non_tie"],
                    "num_candidate_metrics": r["num_candidate_metrics"],
                    "num_verified_metrics": r["num_verified_metrics"],
                    "best_accuracy": (r["verified_metrics"][0]["accuracy"]
                                      if r["verified_metrics"] else 0),
                    "metrics": [
                        {
                            "metric_name": m["metric_name"],
                            "metric_description": m["metric_description"],
                            "accuracy": m["accuracy"],
                            "correct_count": m["correct_count"],
                            "total_verified": m["total_verified"],
                        }
                        for m in r["verified_metrics"]
                    ]
                }
                for r in grand_results
            ],
            "elapsed_seconds": round(time.time() - start_time, 1),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印最终汇总
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  🎯 全部完成!")
        print(f"{'='*60}")
        print(f"  处理组数:        {len(grand_results)}")
        print(f"  生成 metric 数:  {total_metrics}")
        print(f"  平均准确率:      {avg_accuracy:.1%}")
        print(f"  总耗时:          {elapsed:.1f}s")
        print(f"")
        print(f"  📂 输出目录: {OUTPUT_DIR}/")
        print(f"  📋 汇总文件: {summary_file}")
        print(f"")

        # 打印每组最佳 metric
        print(f"  {'维度':<20s} {'组标签':<25s} {'最佳准确率':>10s} {'metric数':>8s}")
        print(f"  {'─'*65}")
        for r in grand_results:
            best = (r["verified_metrics"][0]["accuracy"]
                    if r["verified_metrics"] else 0)
            print(f"  {r['dimension_id']:<20s} "
                  f"{r['group_label']:<25s} "
                  f"{best:>9.1%} "
                  f"{r['num_verified_metrics']:>7d}")


if __name__ == "__main__":
    main()