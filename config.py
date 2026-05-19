"""Shared project configuration."""

import os
from pathlib import Path


# API / model settings
API_KEY_PATH = Path(os.getenv("API_KEY_PATH", r"E:\homework\ai\apikey.txt"))
BASE_URL = os.getenv("BASE_URL", "https://yeysai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


# Proxy settings. Set HTTP_PROXY/HTTPS_PROXY in the environment to override.
HTTP_PROXY = os.getenv("HTTP_PROXY", "http://127.0.0.1:7897")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "http://127.0.0.1:7897")


# Data / output paths
DATASET_PATH = os.getenv("DATASET_PATH", "data/chatbot_arena")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
CLUSTER_OUTPUT_DIR = os.getenv("CLUSTER_OUTPUT_DIR", os.path.join(RESULTS_DIR, "clusters"))
METRICS_OUTPUT_DIR = os.getenv("METRICS_OUTPUT_DIR", os.path.join(RESULTS_DIR, "metrics"))
JUDGMENT_OUTPUT_DIR = os.getenv("JUDGMENT_OUTPUT_DIR", os.path.join(RESULTS_DIR, "judgments"))


# Runtime defaults
SEED = int(os.getenv("SEED", "42"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "500"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))


def apply_proxy_settings():
    """Apply proxy settings before constructing clients or downloading data."""
    if HTTP_PROXY:
        os.environ["HTTP_PROXY"] = HTTP_PROXY
    if HTTPS_PROXY:
        os.environ["HTTPS_PROXY"] = HTTPS_PROXY


def get_api_key():
    """Read the API key from the configured path."""
    return API_KEY_PATH.read_text(encoding="utf-8").strip()


def create_openai_client():
    """Create an OpenAI-compatible client using shared settings."""
    from openai import OpenAI

    apply_proxy_settings()
    return OpenAI(api_key=get_api_key(), base_url=BASE_URL)


def ensure_directories():
    """Create all project output directories."""
    for path in (RESULTS_DIR, CLUSTER_OUTPUT_DIR, METRICS_OUTPUT_DIR, JUDGMENT_OUTPUT_DIR):
        os.makedirs(path, exist_ok=True)


ensure_directories()
