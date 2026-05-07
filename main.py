from __future__ import annotations

import logging
import os
from typing import List

from tqdm import tqdm

from core.data_loader import PairwiseSample, load_ultrafeedback_samples
from core.pipeline import EvaluatorConfig, EvaluatorPipeline
from llm_api import agent_real
from llm_api.agent_real import RealDualAgent


# 告诉 Python 走本地代理软件
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"  # 请把 7890 换成你的代理端口
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897" # 请把 7890 换成你的代理端口
BASE_URL = "https://yeysai.com/v1"
MODEL = "gpt-4o-mini"
TRAIN_SIZE = 30
TEST_SIZE = 90
RANDOM_SEED = 42
key_path=r"E:\homework\ai\apikey.txt"

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_pipeline(api_key: str) -> EvaluatorPipeline:
    config = EvaluatorConfig(random_seed=RANDOM_SEED)
    real_agent = RealDualAgent(
        api_key=api_key,
        base_url=BASE_URL,
        model=MODEL,
        reasons_per_side=config.reasons_per_side,
        rng=None,
    )
    return EvaluatorPipeline(config=config, agent=real_agent)


def train(pipeline: EvaluatorPipeline, train_samples: List[PairwiseSample]) -> None:
    for sample in tqdm(train_samples, desc="Train", unit="pair"):
        if sample.label == "A":
            chosen = sample.answer_a
            rejected = sample.answer_b
        else:
            chosen = sample.answer_b
            rejected = sample.answer_a

        pipeline.train_step(
            topic=sample.topic,
            ans_chosen=chosen,
            ans_rejected=rejected,
            user_prompt=sample.user_prompt,
        )


def evaluate(pipeline: EvaluatorPipeline, test_samples: List[PairwiseSample]) -> int:
    correct = 0
    for sample in tqdm(test_samples, desc="Test", unit="pair"):
        result = pipeline.inference_step(
            topic=sample.topic,
            ans_a=sample.answer_a,
            ans_b=sample.answer_b,
            user_prompt=sample.user_prompt,
        )
        prediction = str(result["winner"])
        if prediction == sample.label:
            correct += 1
    return correct


def print_report(test_size: int, correct: int, metric_count: int) -> None:
    accuracy = correct / test_size if test_size else 0.0
    print("=========================================")
    print("\u5b9e\u9a8c\u7ed3\u679c\u62a5\u544a")
    print(f"\u6d4b\u8bd5\u6837\u672c\u6570: {test_size}\u5bf9")
    print(f"\u6b63\u786e\u9884\u6d4b\u6570: {correct}\u5bf9")
    print(f"\u51c6\u786e\u7387 (Accuracy): {accuracy * 100:.2f}%")
    print(f"\u603b Token \u6d88\u8017: {agent_real.total_tokens}")
    print(f"\u6570\u636e\u5e93\u6700\u7ec8 Metric \u6570\u91cf: {metric_count}\u6761")
    print("=========================================")


def main() -> None:
    with open(key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    if not api_key:
        raise RuntimeError("Please set YEYSAI_API_KEY before running this experiment.")

    logging.info("Loading HuggingFaceH4/ultrafeedback_binarized split=train_prefs ...")
    train_samples, test_samples = load_ultrafeedback_samples(
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    pipeline = build_pipeline(api_key=api_key)
    train(pipeline=pipeline, train_samples=train_samples)
    correct = evaluate(pipeline=pipeline, test_samples=test_samples)
    print_report(test_size=len(test_samples), correct=correct, metric_count=len(pipeline.db))


if __name__ == "__main__":
    configure_logging()
    main()
