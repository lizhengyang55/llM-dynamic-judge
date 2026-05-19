# LLM Metric Database

This project builds a metric database with an LLM, then uses that database to
help an LLM judge which Chatbot Arena response is better.

## Structure

```text
.
|-- download.py                  # Dataset download script
|-- cluster_by_dimensions.py     # Five-dimensional clustering
|-- generate_metrics.py          # Metric-generation interface / implementation slot
|-- judge_with_metrics.py        # Classification -> retrieval -> judgment
|-- main.py                      # Main build/evaluate pipeline
|-- config.py                    # Shared API, model, proxy, and path config
`-- results/
    |-- clusters/
    |-- metrics/
    `-- judgments/
```

## Configuration

Defaults are defined in `config.py`:

- API key path: `E:\homework\ai\apikey.txt`
- Base URL: `https://yeysai.com/v1`
- Model: `gpt-4o-mini`
- Dataset path: `data/chatbot_arena`
- Outputs: `results/clusters`, `results/metrics`, `results/judgments`

You can override them with environment variables:

```powershell
$env:API_KEY_PATH="E:\homework\ai\apikey.txt"
$env:BASE_URL="https://yeysai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:DATASET_PATH="data/chatbot_arena"
```

## Run

Download the dataset first if needed:

```powershell
python download.py
```

Build the metric database and evaluate:

```powershell
python main.py
```

Build only:

```powershell
python main.py --build-only
```

Evaluate using existing metrics:

```powershell
python main.py --skip-build
```

Use smaller sample counts for smoke tests:

```powershell
python main.py --train-num 30 --test-num 20
```

## Note About `generate_metrics.py`

The original metric-generation implementation was not present in the provided
source. `generate_metrics.py` currently preserves the required import interface:

```python
OUTPUT_DIR
process_single_group(cluster_file, group_label)
```

Replace the stub with the original implementation before running the build
phase.
