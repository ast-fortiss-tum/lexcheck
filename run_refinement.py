#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
import random
import glob
from datetime import datetime
from transformers import AutoTokenizer
from src.utils import setup_logging, update_status_file, load_jsonl
from src.data_utils import load_split_data
from src.models import train_model, prepare_model_inputs, evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_aft_training(exp_dirs: list[Path],
                     nli_threshold: float=0.1,
                     ppl_diff_threshold: float=0.0,
                     batch_size: int = 32,
                     num_epochs: int = 1):
    # Initialize lists to accumulate data from all experiment directories
    all_training_adversarial = []
    # Store configs to ensure consistency or for reference
    configs = []

    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        # Load original config
        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)
        configs.append(config)

        task = config["task"]
        model_name = config["model"]
        model_key = config["model_key"]
        foundation_dir = Path(config["foundation_dir"])
        is_generative = "_gen" in model_key

        print(f"Processing experiment directory: {exp_dir.name}")

        # 1. Load attempts and filter by NEW threshold
        attempts_file = exp_dir / "probes" / "attempts.jsonl"
        failures_file = exp_dir / "probes" / "failures.jsonl"
        if not attempts_file.exists() or not failures_file.exists():
            raise FileNotFoundError(f"Attempts file not found at {attempts_file} or {failures_file}. Run probe first.")

        attempts = load_jsonl(attempts_file)
        # Filter: must be flipped AND NLI scores must meet the NEW threshold
        current_accepted_edits = []
        for a in attempts:
            if a.get("flipped") and not a.get("generator_rejected"):
                f_score = a["nli_scores"]["forward"]
                b_score = a["nli_scores"]["backward"]
                if f_score >= nli_threshold and b_score >= nli_threshold:
                    # We need to find the edited text. Since attempts only stores IDs,
                    # we'll look it up in failures.jsonl or reconstruct if necessary.
                    # Simplified here: assuming you want to use the already "accepted" items that pass this specific threshold.
                    current_accepted_edits.append(a)

        if not current_accepted_edits:
            print(f"⚠ No samples found for threshold {nli_threshold}. Skipping.")
            continue

        # Load the actual edited content from failures.jsonl (which contains full objects)

        full_failures = load_jsonl(failures_file)
        # Map by item_id + word to get the "edited" object and log_ppl_diff
        content_map = {(f["original"]["id"], f["word"]): (f["edited"], f.get("log_ppl_diff", float('inf'))) for f in
                       full_failures}

        current_training_adversarial = []
        for a in current_accepted_edits:
            key = (a["item_id"], a["word"])
            if key in content_map:
                edited_content, log_ppl_diff = content_map[key]
                # Filter by ppl_diff_threshold
                if log_ppl_diff < ppl_diff_threshold:
                    current_training_adversarial.append(edited_content)
        print(f"Collected {len(current_training_adversarial)} adversarial samples for {exp_dir}")
        all_training_adversarial.extend(current_training_adversarial)

    # After collecting data from all exp_dirs, check if we have any data to train
    if not all_training_adversarial:
        print("No adversarial samples collected from any of the provided experiment directories. Exiting AFT training.")
        return
    if not configs: # If no configs were loaded (e.g., all exp_dirs failed)
        print("No valid experiment directories processed. Exiting AFT training.")
        return

    # Use parameters from the first successfully loaded config
    main_config = configs[0]
    task = main_config["task"]
    model_key = main_config["model_key"]
    foundation_dir = Path(main_config["foundation_dir"])
    is_generative = "_gen" in model_key
    model_name= str(foundation_dir / "models" / task / model_key)

    first_exp_dir = exp_dirs[0]
    logger = setup_logging(first_exp_dir / "logs")
    logger.info(f"Starting AFT Training for combined exp_dirs with NLI Threshold: {nli_threshold}")

    # 2. Load foundation data
    fit_items = load_jsonl(foundation_dir / "data" / task / "fit.jsonl")
    mining_items = load_jsonl(foundation_dir / "data" / task / "mining.jsonl")

    # Combine original training data
    original_items =  fit_items + mining_items

    # Mix original and adversarial in 1:1 ratio
    min_count = min(len(original_items), len(all_training_adversarial))
    sampled_original = random.sample(original_items, min_count) if len(original_items) > min_count else original_items
    sampled_adversarial = random.sample(all_training_adversarial, min_count) if len(
        all_training_adversarial) > min_count else all_training_adversarial

    # Combine and shuffle
    augmented_items = sampled_original + sampled_adversarial
    random.shuffle(augmented_items)

    print(f"\n{'=' * 60}")
    print(f"AFT TRAINING (Threshold: {nli_threshold})")
    print(f"Adversarial samples: {len(all_training_adversarial)}")
    print(f"Total training data: {len(augmented_items)}")
    print(f"{'=' * 60}\n")

    # 3. Train
    aft_dir = foundation_dir / "models"/ task / f"aft_{model_key}_nli_{nli_threshold}_ppl_{ppl_diff_threshold}"
    aft_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    train_encodings, train_labels = prepare_model_inputs(augmented_items, tokenizer, is_generative=is_generative,
                                                         task=task)

    dev_items = load_jsonl(foundation_dir / "data" / task / "dev.jsonl")
    dev_encodings, dev_labels = prepare_model_inputs(dev_items, tokenizer, is_generative=is_generative, task=task)
    adv_dev_items = load_split_data(foundation_dir, task, "adv_dev")
    adv_dev_encodings, adv_dev_labels = prepare_model_inputs(adv_dev_items, tokenizer, is_generative=is_generative, task=task)

    if is_generative:
        from src.models import GenerativeDataset, train_generative_model
        training_result = train_generative_model(
            model_name=model_name,
            train_dataset=GenerativeDataset(train_encodings, train_labels),
            eval_dataset= GenerativeDataset(dev_encodings, dev_labels),#GenerativeDataset(dev_encodings, dev_labels),GenerativeDataset(adv_dev_encodings, adv_dev_labels),
            output_dir=aft_dir,
            tokenizer=tokenizer,
            num_epochs=num_epochs,
            use_lora=True,  # LoRA integrated
            learning_rate=1e-4, # LoRA requires a larger learning rate (original default is 2e-5)
            batch_size=16,      # Appropriately reduce batch size on single GPU
            #gradient_accumulation_steps=2 # Reduce accumulation steps to increase update frequency
        )
    else:
        from src.models import TextDataset
        num_labels = len(set(train_labels + dev_labels))
        training_result = train_model(
            model_name=model_name,
            train_dataset=TextDataset(train_encodings, train_labels),
            #eval_dataset=TextDataset(dev_encodings, dev_labels),
            eval_dataset= TextDataset(adv_dev_encodings, adv_dev_labels),#=TextDataset(adv_dev_encodings, adv_dev_labels)
            output_dir=aft_dir,
            num_labels=num_labels,
            num_epochs=num_epochs,
            use_lora=True,  # LoRA integrated
            learning_rate=2e-4, #2e-4, # LoRA requires a larger learning rate (original default is 2e-5)
            batch_size=16,      # Appropriately reduce batch size on single GPU
            gradient_accumulation_steps=2 # Reduce accumulation steps to increase update frequency
        )
    
    dev_items = load_split_data(foundation_dir, task, "dev")
    adv_dev_items = load_split_data(foundation_dir, task, "adv_dev")

    # Evaluate old (foundation) model
    print(f"\n{'=' * 60}")
    print(f"EVALUATING OLD MODEL (Foundation) {model_name}")
    print(f"{'=' * 60}\n")

    old_dev_results = evaluate_model(Path(model_name), dev_items, task, is_generative=is_generative,
                                     batch_size=32)
    old_adv_dev_results = evaluate_model(Path(model_name), adv_dev_items, task, is_generative=is_generative,
                                     batch_size=32)
    print(model_name)
    print(f"Old Model - Dev: {old_dev_results['accuracy']}, Adv Dev: {old_adv_dev_results['accuracy']}")

    # Evaluate new (AFT) model
    # print(f"\n{'=' * 60}")
    # print(f"EVALUATING NEW MODEL (AFT) {aft_dir}")
    # print(f"{'=' * 60}\n")

    new_dev_results = evaluate_model(aft_dir, dev_items, task, is_generative=is_generative,
                                     batch_size=32)
    new_adv_dev_results = evaluate_model(aft_dir, adv_dev_items, task, is_generative=is_generative,
                                         batch_size=32)

    print(f"New Model - Dev: {new_dev_results['accuracy']}")
    print(f"New Model - Adv Dev: {new_adv_dev_results['accuracy']}")



    # Create new result entry
    new_entry = {
        "old_dev_acc": old_dev_results['accuracy'],
        "old_adv_dev_acc": old_adv_dev_results['accuracy'],
        "new_dev_acc": new_dev_results['accuracy'],
        "new_adv_dev_acc": new_adv_dev_results['accuracy'],
        "acc_increment": new_dev_results['accuracy'] - old_dev_results['accuracy'],
        "adv_acc_increment": new_adv_dev_results['accuracy'] - old_adv_dev_results['accuracy'],
        "adversarial_count": len(all_training_adversarial),
        "timestamp": datetime.now().isoformat(),
    }

    return new_entry
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g., sst2")
    parser.add_argument("--model-key", type=str, required=True, help="Model key, e.g., distilbert_base_uncased")
    parser.add_argument("--xai", type=str, required=True, help="XAI method, e.g., occlusion")
    parser.add_argument("--operations", type=str, nargs='+', default=["lm", "random"], help="Operations to find")

    parser.add_argument("--nli-threshold", type=float, default=0.95, help="NLI Threshold for selecting edits")
    parser.add_argument("--ppl-diff-threshold", type=float, default=0.0, help="PPL difference threshold for selecting edits")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1)


    debug_inject_path = "experiments/sst2_inject_distilbert_base_uncased_occlusion_lm_0.0_20260212_224402"
    debug_ablate_path = "experiments/sst2_ablate_distilbert_base_uncased_occlusion_lm_0.0_20260212_234616"

    debug_input = [
        "--task", "sst2",
        "--model-key", "Qwen_Qwen2.5_0.5B_gen" ,#"distilbert_base_uncased" "Qwen_Qwen2.5_0.5B_gen" "Qwen_Qwen2.5_0.5B" "facebook_bart_large"
        "--xai", "occlusion",
        "--operations", "random",
        "--nli-threshold", "0.08",
        "--ppl-diff-threshold", "0.3",

        "--batch-size", "32",
        "--epochs", "30"

    ]

    args = parser.parse_args(debug_input)
    base_experiments_dir = "experiments"
    exp_dirs_paths = []

    for op in args.operations:
        for mutator in ["inject", "ablate"]:
        # Build matching pattern, e.g.: experiments/sst2_inject_distilbert_base_uncased_occlusion_*
            pattern = f"{args.task}_{mutator}_{args.model_key}_{args.xai}_{op}*"
            search_path = os.path.join(base_experiments_dir, pattern)
            matches = glob.glob(search_path)
            if not matches:
                print(f"⚠ Warning: No directory found matching pattern: {search_path}")
                continue

            # Sort by timestamp at the end of folder name, take the latest one
            matches.sort()
            latest_dir = matches[-1]
            print(f"🔍 Found latest [{op}] dir: {latest_dir}")
            exp_dirs_paths.append(Path(latest_dir))

    if not exp_dirs_paths:
        raise ValueError("❌ No valid experiment directories found. Please check your parameters.")
    new_entry = run_aft_training(exp_dirs_paths,
                     nli_threshold=args.nli_threshold,
                     ppl_diff_threshold=args.ppl_diff_threshold,
                     batch_size=args.batch_size,
                     num_epochs=args.epochs)
    new_entry.update(vars(args))
    # Auto-record results to JSON file
    results_log_file = Path("model_refinement_results.json")

    # Load existing results if file exists
    if results_log_file.exists():
        with open(results_log_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Check if this model already exists and update, otherwise append
    model_exists = False
    for i, entry in enumerate(all_results):
        if entry.get("model_name") == args.model_key and entry.get("epochs") == args.epochs:
            all_results[i] = new_entry
            model_exists = True
            print(f"✓ Updated existing entry for {args.model_key} with {args.epochs} epochs")
            break

    if not model_exists:
        all_results.append(new_entry)
        print(f"✓ Added new entry for {args.model_key} with {args.epochs} epochs")

    # Save back to file
    with open(results_log_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Results logged to {results_log_file}")
