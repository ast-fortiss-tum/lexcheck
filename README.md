# Explanation-Guided Metamorphic Testing of Specialized Language Models: An Empirical Study

A framework for label-efficient learning through explainability-guided error mining and targeted model editing.

## Overview

**Foundation Phase** (Run Once)
- Global data splitting and supervised fine-tuning (SFT)
- Reusable across all experiments
- Produces baseline models and mining data splits

**Experiment Phase** (Run Many Times)  
- Targeted error mining with different explainers (IG, attention, occlusion)
- Rapid iteration on strategies, thresholds, and model variants

** Refinement Phase** 
- Adaptive fine-tuning (AFT) with learned edits

## Quick Start

### 1. Setup Foundation

```bash
# Run the training setup script to initialize foundation
bash train_setup.sh

# Or manually setup specific task/model combinations
python setup.py --tasks sst2 --models distilbert-base-uncased --xai ig,occlusion,attn --max-items 2000

# For debugging with small dataset
python setup.py --tasks sst2 --models distilbert-base-uncased --max-items 200
```

The `train_setup.sh` script automates foundation setup for multiple task/model combinations (SST2, News, GitHub datasets with various models).

This creates:
```
foundation/
├── data/
│   ├── sst2/
│   │   ├── fit.jsonl      # Training data (90%)
│   │   ├── mining.jsonl   # Mining data (10%)
│   │   ├── dev.jsonl      # Development data
│   │   └── test.jsonl     # Test data
│   └── qnli/
├── models/
│   ├── sst2/
│   │   └── distilbert_base_uncased/  # Trained SFT model
│   └── qnli/
├── pools/                 # Attribute-based edit pools
├── results/               # Analysis outputs
└── config.json           # Foundation config
```

### 2. Run Focused Experiments

```bash
# Run parameter grid search (configured in run_experiments.sh)
bash run_experiments.sh

# Or run single experiment manually
python run_test_generation.py \
    --task sst2 \
    --model distilbert-base-uncased \
    --explainer ig \
    --strategy lm \
    --threshold 0.95
```

Edit `run_experiments.sh` to configure:
- `MODELS`: Model architectures to test
- `EXPLAINERS`: Explainer types (ig, attn, occlusion, random)
- `STRATEGIES`: Placement strategies (prefix, lm, random)
- `OPERATIONS`: inject or ablate operations
- `THRESHOLD`: NLI confidence threshold
- `TASK`: Task name (sst2, news, github)

### 3. Analyze Results

```bash
# View experiment summary
python run_test_generation.py \
    --task sst2 \
    --model distilbert-base-uncased \
    --explainer ig \
    --strategy lm \
    --threshold 0.95 \
    --steps mine,probe

# Analysis tools (see analysis/ folder)
python analysis/02_evaluate_sft.py
python analysis/03_analyze_pools.py
python analysis/visualize_thresholds.py
```

## Complete Workflow

### Foundation Setup (Once)
```bash
# Setup for paper experiments
python setup.py \
    --tasks sst2,qnli \
    --models distilbert-base-uncased,roberta-base \
    --mining-split-ratio 0.1 \
    --seed 42
```

### Parameter Sweep (Many Times)
```bash
# IG explainer experiments - threshold sensitivity
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.99
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.95
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy lm --threshold 0.90

# Strategy comparison
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy random --threshold 0.95
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer ig --strategy prefix --threshold 0.95

# Attention explainer experiments  
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer attn --strategy lm --threshold 0.95
python run_test_generation.py --task sst2 --model distilbert-base-uncased --explainer attn --strategy random --threshold 0.95

# Cross-model transfer
python run_test_generation.py --task sst2 --model roberta-base --explainer ig --strategy lm --threshold 0.95
```

### Analysis
```bash
# Generate comprehensive comparison
python compare_experiments.py

# Analyze edit pools
python analysis/03_analyze_pools.py

# Visualize thresholds and transfer
python analysis/analyze_thresholds.py
python analysis/visualize_transfer.py
```

## Script Reference

**Foundation Setup**
- `setup.py` - Initialize foundation with data splits and baseline models

**Experiments**
- `run_test_generation.py` - Main experiment runner (mine → probe → ablate/inject)
- `run_lm_injection.py` - Language model-based edit injection
- `run_refinement.py` - Refinement and post-processing

## Directory Structure (Git-Tracked)

```
Artifact/
├── src/                        # Core package source code
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_utils.py           # Data loading & preprocessing
│   ├── models.py               # Model training & inference
│   ├── mining.py               # Error mining pipeline
│   ├── mutator.py              # Edit generation & application
│   └── utils.py                # Utility functions
├── train_setup.sh              # Foundation setup automation
├── run_experiments.sh           # Parameter grid search automation
├── setup.py                    # Foundation initialization script
├── run_test_generation.py      # Main experiment runner
├── run_lm_injection.py         # LM-based edit injection
├── run_refinement.py           # Refinement & post-processing
├── pyproject.toml              # Project configuration
├── requirement.txt             # Python dependencies
└── README.md                   # This file

# Note: The following directories are in .gitignore (not tracked)
# foundation/        - SFT baselines and data splits (generated by setup.py)
# experiments/       - Experiment outputs and logs
# analysis/          - Analysis scripts & visualization outputs
# training/          - Training utilities and evaluation scripts
# scripts/           - Helper & utility scripts
# legacy/            - Previous framework versions
# Ollama/            - Local LLM service (optional)
```
