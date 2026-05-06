from __future__ import annotations

import logging
from typing import Sequence

from core.pipeline import EvaluatorPipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def mock_answer(topic: str, quality: str, idx: int) -> str:
    return f"[{topic}] mock {quality} answer #{idx}: contains controlled synthetic signals."


def run_demo(topics: Sequence[str]) -> None:
    pipeline = EvaluatorPipeline()
    pipeline.bootstrap_mock_data(topics=topics, per_topic=4)

    logging.info("========== Evolution / Training ==========")
    for step in range(20):
        topic = topics[step % len(topics)]
        pipeline.train_step(
            topic=topic,
            ans_chosen=mock_answer(topic, "chosen", step),
            ans_rejected=mock_answer(topic, "rejected", step),
        )

        if (step + 1) % 10 == 0:
            logging.info("========== Purge Check After %d Steps ==========", step + 1)
            pipeline.purge()

    logging.info("========== Inference ==========")
    result = pipeline.inference_step(
        topic="coding",
        ans_a="Answer A: clean implementation with tests and clear error handling.",
        ans_b="Answer B: concise but misses edge cases and has weak test coverage.",
    )

    print("\nFinal Inference Result")
    print(f"m      = {result['m_value']:.4f}")
    print(f"S_A    = {result['S_A']:.4f}")
    print(f"S_B    = {result['S_B']:.4f}")
    print(f"P(A>B) = {result['prob_A']:.4f}")
    print(f"winner = {result['winner']}")


if __name__ == "__main__":
    configure_logging()
    run_demo(topics=["math", "coding", "writing", "reasoning"])
