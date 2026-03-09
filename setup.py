#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from pathlib import Path
import sys
import json
from typing import List
# Split training data
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import ExperimentConfig
from src.data_utils import split_dataset, load_task_data, convert_to_standard_format, load_adversarial_data
from src.models import train_model,train_generative_model, TextDataset, GenerativeDataset, prepare_model_inputs
from src.utils import setup_logging, save_jsonl
from src.mining import mine_errors

def setup_foundation(tasks: List[str], models: List[str],
                     mining_split_ratio: float = 0.1, seed: int = 42,
                     max_items: int = None, force_rebuild: bool = False,
                     is_generative: bool = False,
                     xais: List[str] = ["ig", "occlusion"],
                     without_mining: bool = False,
                     resume_training: bool = False,
                     lora_training: bool = False):
    """
    Build the foundation infrastructure experiments.

    This includes:
    1. Data splitting: training, error-mining, dev, test sets
    2. Training SFT (Supervised Fine-Tuning) models
    3. Mining model errors and building candidate word pools

    Args:
        tasks: List of task names (e.g., ['sst2', 'qnli'])
        models: List of model names (e.g., ['distilbert-base-uncased'])
        mining_split_ratio: Ratio of training data reserved for error mining (default: 0.1)
        seed: Random seed for reproducibility
        max_items: Maximum items per split (for debugging)
        force_rebuild: Force rebuild even if files exist
        is_generative: Whether to use generative SFT (CausalLM) instead of classification
        xais: List of XAI signals to use (default: ['ig', 'occlusion'])
        without_mining: Skip mining step
        resume_training: Resume training from last checkpoint
        lora_training: Use LoRA for model training
    """
    # Only rank 0 should run setup (important for DDP mode)
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return Path("foundation")  # Non-rank-0 processes just return

    base_dir = Path("foundation")
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    
    # Setup logging
    logger = setup_logging(base_dir / "logs")
    
    logger.info("Setting up LexCheck foundation...")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {models}")
    logger.info(f"Mode: generative={is_generative}")

    # Create foundation directories
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data for each task
    for task in tasks:
        logger.info(f"Setting up data for {task}")
        
        task_data_dir = data_dir / task
        task_data_dir.mkdir(exist_ok=True)
        
        # Check if data already exists
        if not force_rebuild and all((task_data_dir / f"{split}.jsonl").exists()
                                     for split in ["train", "mining", "dev", "test"]):
            logger.info(f"Data for {task} already exists, skipping...")
            continue
        
        # Load and split data
        train_data, dev_data, test_data = load_task_data(task, max_items)
        logger.info(f"Original data sizes - Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

        train_indices = list(range(len(train_data)))
        np.random.seed(seed)
        
        
        train_indices, mining_indices = train_test_split(
            train_indices,
            test_size=mining_split_ratio,
            random_state=seed,
            stratify=[train_data[i]["label"] for i in train_indices]
        )
        
        
        # Create splits
        sft_train_data = train_data.select(train_indices)
        mining_data = train_data.select(mining_indices)
        logger.info(f"Split sizes - Train: {len(sft_train_data)}, Mining: {len(mining_data)}")

        # Convert to standard format
        train_items = convert_to_standard_format(sft_train_data, task, train_indices)
        mining_items = convert_to_standard_format(mining_data, task, mining_indices)
        dev_items = convert_to_standard_format(dev_data, task)
        test_items = convert_to_standard_format(test_data, task)
        logger.info(
            f"Converted data sizes - Train: {len(train_items)}, Mining: {len(mining_items)}, Dev: {len(dev_items)}, Test: {len(test_items)}")

        # Save splits
        save_jsonl(task_data_dir / "train.jsonl", train_items)
        save_jsonl(task_data_dir / "mining.jsonl", mining_items)
        save_jsonl(task_data_dir / "dev.jsonl", dev_items)
        save_jsonl(task_data_dir / "test.jsonl", test_items)


        # Load and save adversarial dev set
        logger.info(f"Loading adversarial dev data for {task}")
        adv_dev_data = load_adversarial_data(task, max_items)
        if adv_dev_data is not None:
            adv_dev_items = convert_to_standard_format(adv_dev_data, task)
            save_jsonl(task_data_dir / "adv_dev.jsonl", adv_dev_items)
            logger.info(f"Saved adversarial dev set: {len(adv_dev_items)} items")
        else:
            logger.warning(f"No adversarial data available for {task}")
        
        # Save split statistics
        stats = {
            "task": task,
            "splits": {
                "train": len(train_items),
                "mining": len(mining_items),
                "dev": len(dev_items),
                "test": len(test_items)
            },
            "mining_split_ratio": mining_split_ratio,
            "seed": seed
        }
        
        with open(task_data_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Created {task} splits: train={len(train_items)}, mining={len(mining_items)}, dev={len(dev_items)}, test={len(test_items)}")
    
    # Train SFT models for each task-model combination
    for task in tasks:
        for model_name in models:
            # Check model type compatibility
            is_bert_style = any(name in model_name.lower() for name in ["bert", "roberta", "albert", "electra"])

            if is_bert_style and is_generative:
                print(f"⚠ Warning: {model_name} is a BERT-style model and does not support generative mode. Falling back to discriminative.")
                is_generative = False

            model_key = model_name.replace("/", "_").replace("-", "_")
            if is_generative:
                model_key = model_key + "_gen"
            logger.info(f"Training SFT model: {task} + {model_key}")

            task_model_dir = models_dir / task / model_key

            # Check if model already exists
            resume_path = None
            if not force_rebuild and task_model_dir.exists() and (task_model_dir / "training_args.bin").exists():
                if not resume_training:
                    logger.info(f"Model {task}/{model_key} already exists, skipping...")
                    continue
                else:
                    checkpoints = list(task_model_dir.glob("checkpoint-*"))
                    if checkpoints:
                        # get latest checkpoint
                        resume_path = str(max(checkpoints, key=lambda x: int(x.name.split("-")[-1])))
                        print(f"🕒 Found existing checkpoint, resuming from: {resume_path}")

            task_model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load training data
            task_data_dir = data_dir / task
            with open(task_data_dir / "train.jsonl", "r") as f:
                train_items = [json.loads(line) for line in f]
            with open(task_data_dir / "dev.jsonl", "r") as f:
                dev_items = [json.loads(line) for line in f]
            
            # Prepare datasets
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

            kwargs_for_input = {"max_length": 128, "is_generative": is_generative, "task": task}
            train_encodings, train_labels = prepare_model_inputs(train_items, tokenizer, **kwargs_for_input)
            dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer,  **kwargs_for_input)

            if is_generative:
                from src.models import GenerativeDataset
                train_dataset = GenerativeDataset(train_encodings, train_labels)
                dev_dataset = GenerativeDataset(dev_encodings, dev_labels)
            else:
                train_dataset = TextDataset(train_encodings, train_labels)
                dev_dataset = TextDataset(dev_encodings, dev_labels)

            #print("show first 5 dev examples",dev_dataset[0:5])
            # Train model
            if not is_generative:
                # Determine number of labels
                unique_labels = set(train_labels + dev_labels)
                num_labels = len(unique_labels)

                training_result = train_model(
                    model_name=model_name,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    output_dir=task_model_dir,
                    num_labels=num_labels,
                    seed=seed,
                    resume_from_checkpoint=resume_path,
                )
            else:
                logger.info(f"Training generative model {task}/{model_key}")
                training_result = train_generative_model(
                    model_name=model_name,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    output_dir=task_model_dir,
                    seed=seed,
                    tokenizer=tokenizer,
                    resume_from_checkpoint=resume_path,
                    use_lora=lora_training
                )
            logger.info(f"Completed training {task}/{model_key}")

    if without_mining:
        return base_dir

    # Mine errors with both signals after all models are trained
    logger.info("Starting error mining with both signals...")
    pools_dir = base_dir / "pools"
    pools_dir.mkdir(exist_ok=True)
    
    # Create config for mining with both signals
    mining_config = ExperimentConfig(
        tasks=tasks,
        models=models,
        signals=xais,
        strategies=[],  # Not used for mining
        thresholds=[],  # Not used for mining
        mining_split_ratio=mining_split_ratio,
        seed=seed,
        max_items=max_items,
        use_generative_mode=is_generative
    )
    
    # Run mining for each task/model combination
    mining_results = mine_errors(mining_config, base_dir)
    
    # Save mining results summary
    with open(base_dir / "mining_results.json", "w") as f:
        json.dump(mining_results, f, indent=2)
    
    logger.info("Error mining completed!")
    
    # Save foundation config
    foundation_config = {
        "tasks": tasks,
        "models": models,
        "mining_split_ratio": mining_split_ratio,
        "seed": seed,
        "max_items": max_items
    }
    
    with open(base_dir / "config.json", "w") as f:
        json.dump(foundation_config, f, indent=2)
    
    logger.info("Foundation setup completed!")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Models: {models_dir}")
    
    return base_dir

def main():
    parser = argparse.ArgumentParser(description="Setup LexCheck foundation data and models")
    
    parser.add_argument("--tasks", type=str, default='sst2',# required=True
                       help="Comma-separated list of tasks (e.g., sst2,qnli)")
    parser.add_argument("--models", type=str, default='distilbert-base-uncased', # required=True,
                       help="Comma-separated list of models (e.g., distilbert-base-uncased,roberta-base)")
    parser.add_argument("--mining-split-ratio", type=float, default=0.1,
                       help="Ratio for mining split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--max-items", type=int,
                       help="Max items per split for debugging")
    parser.add_argument("--force", action="store_true",
                       help="Force re-creation even if files exist")
    parser.add_argument("--generative", action="store_true",
                           help="Train generative models (CausalLM) instead of classification")
    parser.add_argument("--xai", type=str, default='ig,occlusion',
                        help="list of XAI signals to use (e.g., ig,occlusion,random,attn)")
    parser.add_argument("--without-mining", action="store_true")
    parser.add_argument("--resume-training", action="store_true",
                        help="Skip mining step")
    parser.add_argument("--lora-training", action="store_true")
    args = parser.parse_args()
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    xais = [x.strip() for x in args.xai.split(",")]
    try:
        setup_foundation(
            tasks=tasks,
            models=models,
            mining_split_ratio=args.mining_split_ratio,
            seed=args.seed,
            max_items=args.max_items,
            force_rebuild=args.force,
            is_generative=args.generative,
            xais=xais,
            without_mining=args.without_mining,
            resume_training=args.resume_training,
            lora_training=args.lora_training,
        )
        print("✓ Foundation setup completed successfully!")
        
    except Exception as e:
        print(f"✗ Foundation setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
    # python setup.py --tasks sst2 --models distilbert-base-uncased --max-items 200 --xai ig,occlusion,attn

