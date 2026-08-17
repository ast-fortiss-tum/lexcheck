# Explanation-Guided Metamorphic Testing of Specialized Language Models: An Empirical Study

A framework combines metamorphic testing, explainability methods, and language
models to generate and evaluate failure-inducing variants for specialized
language models. This repository contains the implementation, experiment
scripts, analysis utilities, and human-study materials used in the paper.

## Overview

The experimental workflow consists of three stages:

1. **Foundation setup**: prepare data splits and fine-tune the systems under
   test.
2. **Test generation**: identify influential input regions and generate
   contextual injection or ablation variants.
3. **Evaluation and analysis**: verify semantic validity, measure attack
   success, and analyze results across datasets, models, and configurations.

The experiments cover SST-2, AG News, and a GitHub issue-classification
dataset. Supported explainers include integrated gradients, attention,
occlusion, and a random baseline.

## Requirements

- Python 3.10 or 3.11
- A Bash-compatible shell for the provided automation scripts
- Sufficient storage and compute capacity for downloading and running the
  evaluated language models

Install the Python package and its dependencies from the repository root:

```bash
pip install -e .
```

## Quick Start

### 1. Prepare the Foundation

Run the automated setup:

```bash
bash train_setup.sh
```

Alternatively, configure a task and model directly:

```bash
python setup.py \
    --tasks sst2 \
    --models distilbert-base-uncased \
    --xai ig,occlusion,attn \
    --max-items 2000
```

For a small debugging run:

```bash
python setup.py \
    --tasks sst2 \
    --models distilbert-base-uncased \
    --max-items 200
```

The setup produces task-specific data splits, fine-tuned baseline models,
edit pools, and configuration files under `foundation/`.

### 2. Generate Tests

Run the configured parameter sweep:

```bash
bash run_experiments.sh
```

Run a single configuration:

```bash
python run_test_generation.py \
    --task sst2 \
    --model distilbert-base-uncased \
    --explainer ig \
    --strategy lm \
    --threshold 0.95
```

The main options are:

- `MODELS`: model architectures under test
- `EXPLAINERS`: `ig`, `attn`, `occlusion`, or `random`
- `STRATEGIES`: `prefix`, `lm`, or `random`
- `OPERATIONS`: injection or ablation
- `THRESHOLD`: semantic-verification threshold
- `TASK`: `sst2`, `news`, or `github`

Edit `run_experiments.sh` to configure a complete sweep.

### 3. Analyze Results

Run selected pipeline stages directly:

```bash
python run_test_generation.py \
    --task sst2 \
    --model distilbert-base-uncased \
    --explainer ig \
    --strategy lm \
    --threshold 0.95 \
    --steps mine,probe
```

Example analysis commands:

```bash
python analysis/02_evaluate_sft.py
python analysis/03_analyze_pools.py
python analysis/visualize_thresholds.py
```

## Reproducing Experiment Variants

### Foundation setup

```bash
python setup.py \
    --tasks sst2,news,github \
    --models distilbert-base-uncased,roberta-base \
    --mining-split-ratio 0.1 \
    --seed 42
```

### Threshold sensitivity

```bash
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.99
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.95
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.90
```

### Strategy comparison

```bash
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy random --threshold 0.95
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy prefix --threshold 0.95
```

### Cross-model evaluation

```bash
python run_test_generation.py --task sst2 --model roberta-base --explainer ig --strategy lm --threshold 0.95
```

## Repository Structure

```text
.
|-- Ollama_chat/              # Local LLM integration
|-- Research Questions/       # Results organized by research question
|-- mturk_survey/             # Human-study materials and results
|-- src/                      # Core implementation
|-- pyproject.toml            # Package metadata and dependencies
|-- run_experiments.sh        # Experiment automation
|-- run_test_generation.py    # Main test-generation entry point
|-- setup.py                  # Foundation setup
|-- train_setup.sh            # Automated training setup
`-- README.md
```

Generated models, downloaded datasets, and experiment outputs may be excluded
from version control because of their size. Consult `.gitignore` before
expecting generated artifacts to appear in the repository.

## Citation

If you use LexCheck or this replication package, please cite:

```bibtex
@InProceedings{chen2026lexcheck,
  author    = {Xingcheng Chen and Mehmet Besenk and Andrea Stocco},
  title     = {Explanation-Guided Metamorphic Testing of Specialized
               Language Models: An Empirical Study},
  booktitle = {20th International Symposium on Empirical Software Engineering
               and Measurement (ESEM 2026)},
  series    = {Leibniz International Proceedings in Informatics (LIPIcs)},
  volume    = {394},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum fuer Informatik},
  year      = {2026},
  note      = {To appear}
}
```

The citation will be updated with the final page range and DOI after the
proceedings are published.

## License

This project is licensed under the [MIT License](LICENSE).
