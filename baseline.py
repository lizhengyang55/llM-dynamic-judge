# baseline_llm_judge.py
import os
import json
import time
import random
from datasets import load_from_disk
from openai import OpenAI
from pathlib import Path

# ===================== 配置区 =====================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

api_key = Path(r"E:\homework\ai\apikey.txt").read_text(encoding="utf-8").strip()
BASE_URL = "https://yeysai.com/v1"
MODEL_NAME = "gpt-4o-mini"

NUM_SAMPLES = 500
SEED = 42

DATASET_PATH = "data/chatbot_arena"
OUTPUT_PATH = "results/baseline_results.json"
# ==================================================

os.makedirs("results", exist_ok=True)
client = OpenAI(api_key=api_key, base_url=BASE_URL)


def build_prompt(conversation_a, conversation_b):
    """构造 prompt，让大模型判断哪个回答更好"""

    def format_conversation(conv):
        text = ""
        for turn in conv:
            role = turn["role"].upper()
            content = turn["content"]
            if len(content) > 2000:
                content = content[:2000] + "...[truncated]"
            text += f"[{role}]: {content}\n"
        return text.strip()

    conv_a_text = format_conversation(conversation_a)
    conv_b_text = format_conversation(conversation_b)

    prompt = f"""You are a fair judge. Compare the two AI assistant responses below.

=== Assistant A ===
{conv_a_text}

=== Assistant B ===
{conv_b_text}

Which is better? You MUST pick one. Reply with ONLY a single letter: A or B.
Do NOT say "tie". Do NOT explain. Do NOT add any other text. Just one letter: A or B."""

    return prompt


def parse_judgment(response_text):
    """解析模型输出，只接受 A 或 B"""
    if not response_text or not response_text.strip():
        return "unknown"

    text = response_text.strip().upper()

    # 精确匹配单字符
    if text in ("A", "A."):
        return "model_a"
    if text in ("B", "B."):
        return "model_b"

    # 取第一个有效字符
    first_char = text[0]
    if first_char == "A":
        return "model_a"
    elif first_char == "B":
        return "model_b"

    # 全文搜索
    text_lower = text.lower()
    if "assistant a" in text_lower and "better" in text_lower:
        return "model_a"
    if "assistant b" in text_lower and "better" in text_lower:
        return "model_b"

    # 最后兜底：找第一个 A 或 B
    for char in text:
        if char == 'A':
            return "model_a"
        elif char == 'B':
            return "model_b"

    return "unknown"


def call_llm(prompt, max_retries=3):
    """调用大模型API，带重试和空响应重试"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a judge. You must pick a winner. Reply with only A or B."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content

            if content is None or content.strip() == "":
                print(f"  [重试 {attempt + 1}/{max_retries}] 空响应，"
                      f"finish_reason={response.choices[0].finish_reason}")
                time.sleep(2)
                continue

            return content.strip()

        except Exception as e:
            print(f"  [重试 {attempt + 1}/{max_retries}] 错误: {e}")
            time.sleep(2 ** attempt)

    return "error"


def normalize_winner(winner_str):
    """统一 ground truth 的 winner 格式"""
    if "model_a" in winner_str:
        return "model_a"
    elif "model_b" in winner_str:
        return "model_b"
    else:
        return "tie"


def main():
    # ========== 测试API ==========
    print("测试API连接...")
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with only the letter A"}],
            max_tokens=64,
        )
        content = resp.choices[0].message.content
        print(f"✅ API正常: [{content}]")
        print(f"   finish_reason: {resp.choices[0].finish_reason}\n")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return

    # ========== 加载数据 ==========
    print("加载数据集...")
    ds = load_from_disk(DATASET_PATH)
    data = ds["train"]
    print(f"总共 {len(data)} 条数据")

    random.seed(SEED)
    indices = random.sample(range(len(data)), min(NUM_SAMPLES, len(data)))
    print(f"采样 {len(indices)} 条进行评测\n")

    # ========== 开始评测 ==========
    results = []
    correct = 0
    total = 0          # 只计 GT 为 A/B 的有效样本
    skipped_tie = 0    # GT 为 tie 跳过的数量
    error_count = 0
    empty_count = 0

    gt_distribution = {"model_a": 0, "model_b": 0, "tie": 0}
    pred_distribution = {"model_a": 0, "model_b": 0, "unknown": 0}

    start_time = time.time()

    for i, idx in enumerate(indices):
        sample = data[idx]

        question_id = sample["question_id"]
        conversation_a = sample["conversation_a"]
        conversation_b = sample["conversation_b"]
        ground_truth = normalize_winner(sample["winner"])
        model_a_name = sample["model_a"]
        model_b_name = sample["model_b"]

        gt_distribution[ground_truth] += 1

        # ---- 跳过 GT 为 tie 的样本 ----
        if ground_truth == "tie":
            skipped_tie += 1
            print(f"[{i + 1}/{len(indices)}] qid={question_id[:12]}..., "
                  f"{model_a_name} vs {model_b_name}, GT=tie → 跳过")
            continue

        print(f"[{i + 1}/{len(indices)}] qid={question_id[:12]}..., "
              f"{model_a_name} vs {model_b_name}, GT={ground_truth}")

        prompt = build_prompt(conversation_a, conversation_b)
        response_text = call_llm(prompt)

        if response_text == "error":
            error_count += 1
            prediction = "unknown"
            print(f"  → API调用失败")
        else:
            if not response_text.strip():
                empty_count += 1

            prediction = parse_judgment(response_text)
            is_correct = prediction == ground_truth
            symbol = "✓" if is_correct else ("?" if prediction == "unknown" else "✗")
            print(f"  → 输出: [{response_text[:60]}] → 预测: {prediction} {symbol}")

            if prediction != "unknown":
                total += 1
                if is_correct:
                    correct += 1

        pred_distribution[prediction] = pred_distribution.get(prediction, 0) + 1

        results.append({
            "index": idx,
            "question_id": question_id,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "raw_response": response_text,
            "is_correct": prediction == ground_truth if prediction != "unknown" else False,
        })

        # 每20条打印中间结果
        if (i + 1) % 20 == 0:
            acc = correct / total * 100 if total > 0 else 0
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(indices) - i - 1) / speed
            print(f"\n{'=' * 50}")
            print(f"  进度: {i + 1}/{len(indices)} (已跳过 {skipped_tie} 条 tie)")
            print(f"  A/B准确率: {correct}/{total} = {acc:.2f}%")
            print(f"  空响应: {empty_count}, API失败: {error_count}")
            print(f"  耗时: {elapsed:.0f}s, 预计剩余: {eta:.0f}s")
            print(f"{'=' * 50}\n")

    # ==================== 最终统计 ====================
    elapsed_total = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("                    最终结果")
    print("=" * 60)
    print(f"  评测模型:          {MODEL_NAME}")
    print(f"  总采样数:          {len(indices)}")
    print(f"  GT为tie跳过:       {skipped_tie}")
    print(f"  实际评测(A/B):     {len(indices) - skipped_tie}")
    print(f"  API调用失败:       {error_count}")
    print(f"  空响应次数:        {empty_count}")
    print(f"  有效判定(A/B):     {total}")
    print(f"  总耗时:            {elapsed_total:.1f}s")
    print(f"")
    print(f"  A/B二选一准确率:   {correct}/{total} = {accuracy:.2f}%")
    print(f"")
    print(f"  GT分布(全部):  {gt_distribution}")
    print(f"  预测分布:      {pred_distribution}")
    print("=" * 60)

    output = {
        "config": {
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "mode": "A/B only (tie samples skipped)",
        },
        "metrics": {
            "accuracy_ab": round(accuracy, 2),
            "correct": correct,
            "total_ab": total,
            "skipped_tie": skipped_tie,
            "error_count": error_count,
            "empty_count": empty_count,
            "elapsed_seconds": round(elapsed_total, 1),
            "gt_distribution": gt_distribution,
            "pred_distribution": pred_distribution,
        },
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()