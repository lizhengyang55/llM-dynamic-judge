"""Download lmsys/chatbot_arena_conversations to data/chatbot_arena."""

import os

from datasets import load_dataset
from huggingface_hub import login

from config import DATASET_PATH, apply_proxy_settings


def main():
    apply_proxy_settings()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("downloading ...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    dataset.save_to_disk(DATASET_PATH)
    print(f"done! saved to {DATASET_PATH}")
    print(dataset)


if __name__ == "__main__":
    main()
