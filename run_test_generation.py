#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import sys
import json
import torch
import random
import numpy as np
import time
from transformers import AutoTokenizer
from datetime import datetime
from typing import Optional
from collections import Counter
from tqdm import tqdm
import pandas as pd
from src.utils import setup_logging, create_run_directory, update_status_file, load_jsonl, save_jsonl
from src.mutator import apply_edit, NLIChecker, FlipTester, LMGuidedMutator, PPLCalculatorCorrect
from src.models import train_model, TextDataset, prepare_model_inputs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ExperimentRunner:
    def __init__(self, task: str, model: str, explainer: str, strategy: str, threshold: float,
                 foundation_dir: Path = Path("foundation"),
                 experiments_dir: Path = Path("experiments"),
                 experiment_name: Optional[str] = None,
                 max_probe_items: Optional[int] = None,
                 candidates_per_item: int = 3,
                 batch_size: int = 32,
                 operation: str = "inject"):

        self.task = task
        self.model = model
        self.model_key = model.replace("/", "_").replace("-", "_")
        self.is_generative = "_gen" in self.model_key

        self.explainer = explainer
        self.max_probe_items = max_probe_items
        self.candidates_per_item = candidates_per_item
        self.batch_size = batch_size
        self.strategy = strategy
        self.threshold = threshold
        self.operation = operation # "inject" or "ablate"

        self.foundation_dir = foundation_dir
        self.experiments_dir = experiments_dir
        
        # Create experiment directory
        if self.is_already_run():
            # Existing experiment, skip
            print(f"Experiment {self.task} for {self.model} with {self.operation} and {self.explainer} already exists. Skipping.")
            return

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{task}_{operation}_{self.model_key}_{explainer}_{strategy}_{threshold}_{timestamp}"
        
        self.experiment_dir = experiments_dir / experiment_name

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.experiment_dir / "logs")
        
        # Validate foundation exists
        self._validate_foundation()
        
        # Save experiment config
        self._save_config()
    
    def _validate_foundation(self):
        """Validate that foundation data and models exist"""
        
        # Check foundation config
        """foundation_config_path = self.foundation_dir / "config.json"
        if not foundation_config_path.exists():
            raise FileNotFoundError(f"Foundation not found. Run setup.py first.")
        
        with open(foundation_config_path, 'r', encoding='utf-8') as f:
            foundation_config = json.load(f)
        
        if self.task not in foundation_config["tasks"]:
            raise ValueError(f"Task {self.task} not in foundation. Available: {foundation_config['tasks']}")
        
        if self.model not in foundation_config["models"]:
            raise ValueError(f"Model {self.model} not in foundation. Available: {foundation_config['models']}")"""
        
        # Check data files
        data_dir = self.foundation_dir / "data" / self.task
        for split in ["train", "mining", "dev", "test"]:
            split_file = data_dir / f"{split}.jsonl"
            if not split_file.exists():
                raise FileNotFoundError(f"Data file missing: {split_file}")
        
        # Check model
        model_dir = self.foundation_dir / "models" / self.task / self.model_key
        if not model_dir.exists() or not (model_dir / "training_args.bin").exists():
            raise FileNotFoundError(f"Model missing: {model_dir}")
        
        self.logger.info("Foundation validation passed")

    def _save_config(self):
        """Save experiment configuration"""
        
        config = {
            "task": self.task,
            "model": self.model,
            "model_key": self.model_key,
            "explainer": self.explainer,
            "strategy": self.strategy,
            "threshold": self.threshold,
            "operation": self.operation,
            "foundation_dir": str(self.foundation_dir),
            "experiment_dir": str(self.experiment_dir),
            "created_at": datetime.now().isoformat()
        }
        
        with open(self.experiment_dir / "config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def is_already_run(self) -> bool:
        """Check if the current configuration already exists in master_results.csv"""
        master_table_path = self.experiments_dir / "master_results.csv"
        if not master_table_path.exists():
            return False
        try:
            df = pd.read_csv(master_table_path)
            # Define key columns for duplicate detection
            query = (
                (df['task'] == self.task) &
                (df['model'] == self.model) &
                (df['operation'] == self.operation) &
                (df['explainer'] == self.explainer) &
                (df['strategy'] == self.strategy) &
                (df['nli_threshold'] == self.threshold)
            )
            return query.any()
        except Exception as e:
            self.logger.error(f"Error checking master table: {e}")
            return False

    def mine_errors(self) -> dict:
        """Load mining results from foundation (already mined during setup)"""

        self.logger.info(f"Loading mining results for {self.explainer} explainer from foundation")
        update_status_file(self.experiment_dir, "mine", "⚡", "Loading mining results...")

        try:
            pool_prefix = "pool" if self.operation == "inject" else "correct_pool"
            # Load pre-computed mining results from foundation
            foundation_mined_file = self.foundation_dir / "pools" / self.task / f"mined_errors_{self.model_key}.jsonl"
            foundation_pool_file = self.foundation_dir / "pools" / self.task / f"{pool_prefix}_{self.explainer}_{self.model_key}.jsonl"

            if not foundation_mined_file.exists():
                raise FileNotFoundError(
                    f"Mining results not found in foundation: {foundation_mined_file}. Run setup.py first.")

            if not foundation_pool_file.exists():
                raise FileNotFoundError(f"Pool not found in foundation: {foundation_pool_file}. Run setup.py first.")

            # We use the foundation files directly now, but we still load counts for the summary
            misclassified_items = load_jsonl(foundation_mined_file)
            candidates = load_jsonl(foundation_pool_file)

            # Compute statistics
            all_words = [c.get("word", c.get("token", "")) for c in candidates]
            word_counts = Counter(all_words)

            result = {
                "mining": {
                    "num_misclassified": len(misclassified_items),
                    "source": "foundation (pre-computed)"
                },
                "pool": {
                    "pool_size": len(candidates),
                    "unique_words": len(set(all_words)),
                    "top_10_words": word_counts.most_common(10),
                },
                "pool_file": str(foundation_pool_file)
            }

            # Save mining results
            with open(self.experiment_dir / "mining_results.json", "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            update_status_file(self.experiment_dir, "mine", "✓", f"Loaded {len(misclassified_items)} errors from foundation")

            print(f"✓ Loaded {len(misclassified_items)} misclassified items from foundation")
            print(f"  Pool size: {len(candidates)} candidates")
            print(f"  Unique words: {len(set(all_words))}")
            print(f"  Top words: {', '.join([f'{w}({c})' for w, c in word_counts.most_common(5)])}")

            return result

        except Exception as e:
            update_status_file(self.experiment_dir, "mine", "✗", f"Failed: {str(e)}")
            raise
    
    def probe_edits(self) -> dict:
        """Run probing with specific strategy and threshold"""
        
        self.logger.info(f"Probing with {self.strategy} strategy, threshold {self.threshold}")
        update_status_file(self.experiment_dir, "probe", "⚡", f"Probing {self.strategy}@{self.threshold}...")
        
        try:
            # Create probes directory
            probes_dir = self.experiment_dir / "probes"
            probes_dir.mkdir(exist_ok=True)
            start_time = time.time()
            # Run custom probing
            result = self._probe_edits_foundation()

            duration = time.time() - start_time
            result["results"]["duration_seconds"] = duration

            # Save probe results
            with open(self.experiment_dir / "probe_results.json", "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            success_rate = result["results"]["overall_success_rate"]
            update_status_file(self.experiment_dir, "probe", "✓", f"Success rate: {success_rate:.3f}")

            self._log_to_master_table(result)
            return result
            
        except Exception as e:
            update_status_file(self.experiment_dir, "probe", "✗", f"Failed: {str(e)}")
            raise

    def _log_to_master_table(self, probe_result: dict):
            """Log experiment results, including all key statistics selected in Terminal"""
            master_table_path = self.experiments_dir / "master_results.csv"

            attempts_file = Path(probe_result["files"]["attempts"])
            if not attempts_file.exists():
                return

            attempts_data = load_jsonl(attempts_file)
            total_proposals = len(attempts_data)
            gen_rejections = sum(1 for a in attempts_data if a.get("generator_rejected"))

            flipped_items = [a for a in attempts_data if a.get("flipped")]
            nli_pass_count = sum(1 for a in flipped_items if a.get("accepted"))

            # Metric calculation (for flipped items)
            nli_pass_rate_on_flipped = (nli_pass_count / len(flipped_items)) if flipped_items else 0.0
            avg_ppl_on_flipped = np.mean([a["ppl_diff"] for a in flipped_items]) if flipped_items else 0.0

            # Group statistics (max_neg / max_pos)
            group_stats = {}
            for ct in ["max_neg", "max_pos"]:
                group = [a for a in attempts_data if a.get("cand_type") == ct]
                g_total = len(group)
                g_flipped = sum(1 for a in group if a.get("flipped"))
                g_accepted = sum(1 for a in group if a.get("accepted") and a.get("flipped"))
                group_stats[f"{ct}_total"] = g_total
                group_stats[f"{ct}_flip_rate"] = g_flipped / g_total if g_total > 0 else 0.0
                group_stats[f"{ct}_success_rate"] = g_accepted / g_total if g_total > 0 else 0.0

            new_row = {
                "task": self.task,
                "model": self.model,
                "operation": self.operation,
                "explainer": self.explainer,
                "strategy": self.strategy,
                "nli_threshold": self.threshold,
                # --- Terminal selected core values ---
                "total_proposals": total_proposals,
                "gen_rejections": gen_rejections,
                "flipped_count": len(flipped_items),
                "nli_accepted_on_flipped": nli_pass_count,
                "overall_success_rate": probe_result["results"]["overall_success_rate"], # Already filtered by NLI and Flipped
                # --- PPL & NLI Pass Rate ---
                "nli_pass_rate_flipped": nli_pass_rate_on_flipped,
                "avg_ppl_increase_flipped": avg_ppl_on_flipped,
                # --- Group values ---
                **group_stats,
                "duration_seconds": probe_result["results"].get("duration_seconds", 0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            df = pd.DataFrame([new_row])
            if master_table_path.exists():
                # Check for new columns, update header if necessary
                old_df = pd.read_csv(master_table_path, nrows=0)
                if set(df.columns) != set(old_df.columns):
                    df.to_csv(master_table_path, mode='a', header=True, index=False)
                else:
                    df.to_csv(master_table_path, mode='a', header=False, index=False)
            else:
                df.to_csv(master_table_path, index=False)

            self.logger.info(f"Master results expanded and saved to {master_table_path}")

    def _probe_edits_foundation(self) -> dict:
        """Run probing using foundation structure directly"""

        # 1. Setup paths and tools
        pool_file = self.foundation_dir / "pools" / self.task / f"{'pool' if self.operation == 'inject' else 'correct_pool'}_{self.explainer}_{self.model_key}.jsonl"
        correct_file = self.foundation_dir / "pools" / self.task / f"correctly_classified_{self.model_key}.jsonl"
        model_path = self.foundation_dir / "models" / self.task / self.model_key

        candidates = load_jsonl(pool_file)
        correct_items = load_jsonl(correct_file)
        items_by_id = {item["id"]: item for item in correct_items}

        nli_checker = NLIChecker(model_name="cross-encoder/nli-deberta-v3-small")
        flip_tester = FlipTester(model_path)
        ppl_calculator = PPLCalculatorCorrect(model_id="gpt2", device="cuda")  # Sliding window PPL calculator
        lm_injector = LMGuidedMutator() if self.strategy == "lm" else None


        attempts, failures = [], []


        # 2. Logic for "Ablation" mode often requires pairing items with their specific critical words
        # In injection, we pick random candidates. In ablation, we use the item's own tokens.
        probe_queue = []
        if self.operation == "ablate":
            # Ablation uses candidates which are words from the items themselves
            for cand in candidates:

                item_id = cand.get("item_id")
                if item_id in items_by_id:
                    cand_type = cand.get("type")  # max_pos or max_neg
                    probe_queue.append((items_by_id[item_id], cand.get("word"), cand_type))
        else:
            # Injection uses correctly classified items and random candidate words
            # Group by word and retain metadata
            unique_cand_map = {}
            for c in candidates:
                word = c.get("word", c.get("token", ""))
                if word not in unique_cand_map:
                    unique_cand_map[word] = c


            unique_words = list(unique_cand_map.keys())
            unique_candidates = list(unique_cand_map.values())
            self.logger.info(f"Injection pool processed: {len(candidates)} raw -> {len(unique_words)} unique candidates")

            # Injection: uniformly distribute unique candidates
            random.seed(42)
            max_items = self.max_probe_items or len(correct_items)

            shuffled_unique_cands = unique_candidates.copy()
            random.shuffle(shuffled_unique_cands)
            cand_idx = 0

            for item in correct_items[:max_items]:
                for _ in range(self.candidates_per_item):
                    # cand = random.choice(candidates)
                    # Injection logic modified: balance distribution of unique candidates
                    cand = shuffled_unique_cands[cand_idx % len(shuffled_unique_cands)]
                    probe_queue.append((item, cand.get("word"), cand.get("type")))
                    cand_idx += 1

        pbar = tqdm(total=len(probe_queue), desc=f"Probing ({self.operation})")

        # 3. Batch processing logic
        word_to_tasks = {}
        for task_item in probe_queue:
            word = task_item[1]
            if word not in word_to_tasks:
                word_to_tasks[word] = []
            word_to_tasks[word].append(task_item)

        for word, tasks in word_to_tasks.items():
            # For LM strategy, use a fixed small batch (e.g., 4)
            lm_batch_size = 3 if self.strategy == "lm" else 1 #

            for i in range(0, len(tasks), lm_batch_size):
                batch_tasks = tasks[i: i + lm_batch_size]
                current_batch_edits = []
                valid_meta = []

                if self.strategy == "lm":
                    items = [t[0] for t in batch_tasks]
                    # Call LLM once to process multiple sentences with this word
                    res_dict = lm_injector.batch_modify_word_in_sentences(word, items, operation=self.operation)

                    for item, _, cand_type in batch_tasks:
                        edit_res = res_dict.get(item["id"])
                        if edit_res and not edit_res["rejected"]:
                            # make sure final_a is not empty
                            final_a = edit_res["edited_text_a"] if edit_res["edited_text_a"].strip() else "."
                            final_b = edit_res.get("edited_text_b")
                            current_batch_edits.append((item["text_a"], final_a))
                            valid_meta.append({"item": item, "word": word, "cand_type": cand_type,
                                               "edit_a": final_a, "edit_b": final_b})
                        else:
                            attempts.append({"item_id": item["id"], "word": word, "cand_type": cand_type,
                                             "generator_rejected": True,
                                             "accepted": False,
                                             "flipped": False})
                            pbar.update(1)
                else:
                    # Simple strategies (random, prefix) - process one at a time
                    item, word, cand_type = batch_tasks[0]
                    edit_a, edit_b = apply_edit(item["text_a"], item.get("text_b"), word, self.strategy,
                                                operation=self.operation)
                    current_batch_edits.append((item["text_a"], edit_a))
                    valid_meta.append(
                        {"item": item, "word": word, "cand_type": cand_type, "edit_a": edit_a, "edit_b": edit_b})

                if not current_batch_edits:
                    continue

                # 1. First perform flip detection (full detection for this batch)
                flip_orig, flip_edit = [], []
                for idx, meta in enumerate(valid_meta):
                    lbl = meta["item"].get("label") or meta["item"].get("true_label")
                    orig_item = meta["item"].copy()
                    orig_item["label"] = lbl
                    flip_orig.append(orig_item)
                    flip_edit.append({
                        "id": f"{meta['item'].get('id')}_edited",
                        "text_a": meta["edit_a"], "text_b": meta["edit_b"], "label": lbl
                    })

                flips = flip_tester.test_batch_flips(flip_orig, flip_edit)

                # 2. Filter out flipped samples
                flipped_indices = [j for j, f in enumerate(flips) if f["flipped"]]

                # 3. Separately evaluate NLI and PPL on flipped samples
                nli_results_map = {}
                ppl_diffs_map = {}

                if flipped_indices:
                    def get_full_text(ta, tb):
                            return f"{ta} {tb}" if tb else ta
                    flipped_pairs = [(valid_meta[j]["item"]["text_a"], valid_meta[j]["edit_a"]) for j in
                                     flipped_indices]
                    # Separately evaluate NLI
                    nli_batch = nli_checker.check_batch_bidirectional_entailment(flipped_pairs,
                                                                                 threshold=self.threshold)

                    # Separately evaluate PPL (only for flipped samples)
                    orig_texts_subset = [
                        get_full_text(valid_meta[j]["item"]["text_a"], valid_meta[j]["item"].get("text_b")) for j in
                        flipped_indices]
                    edit_texts_subset = [get_full_text(valid_meta[j]["edit_a"], valid_meta[j]["edit_b"]) for j in
                                         flipped_indices]

                    # Additional safeguard: if merged text is still empty
                    orig_texts_subset = [t if t.strip() else "." for t in orig_texts_subset]
                    edit_texts_subset = [t if t.strip() else "." for t in edit_texts_subset]

                    ppl_diffs = ppl_calculator.compute_diff(orig_texts_subset, edit_texts_subset)

                    for i, f_idx in enumerate(flipped_indices):
                        nli_results_map[f_idx] = nli_batch[i]
                        ppl_diffs_map[f_idx] = ppl_diffs[i]

                # 4. Record all attempts
                for idx, flip_res in enumerate(flips):
                    is_flipped = flip_res["flipped"]
                    # NLI and PPL are only valid for flipped, default to False/0.0 for non-flipped
                    nli_info = nli_results_map.get(idx,
                                                   {"passes_gate": False, "forward_score": 0.0, "backward_score": 0.0})
                    ppl_val = ppl_diffs_map.get(idx, 0.0)

                    if is_flipped and nli_info["passes_gate"]:
                        failures.append({
                            "original": valid_meta[idx]["item"],
                            "edited": flip_edit[idx],
                            "word": valid_meta[idx]["word"],
                            "ppl_diff": ppl_val
                        })

                    attempts.append({
                        "item_id": valid_meta[idx]["item"]["id"],
                        "word": valid_meta[idx]["word"],
                        "cand_type": cand_type,
                        "generator_rejected": False,
                        "flipped": is_flipped,
                        "accepted": nli_info["passes_gate"],  # Here 'accepted' only represents NLI result
                        "ppl_diff": ppl_val,
                        "nli_scores": {"forward": nli_info["forward_score"], "backward": nli_info["backward_score"]}
                    })
                    pbar.update(1)

        pbar.close()
        
        # Save results
        probes_dir = self.experiment_dir / "probes"
        attempts_file = probes_dir / "attempts.jsonl"
        failures_file = probes_dir / "failures.jsonl"

        save_jsonl(attempts_file, attempts)
        save_jsonl(failures_file, failures)

        # Load attempts data for analysis
        attempts_data = load_jsonl(attempts_file)

        # Compute summary statistics
        total_proposals = len(attempts)
        generator_rejections = sum(1 for a in attempts if "generator_rejected" in a and a["generator_rejected"])
        accepted_count = sum(1 for a in attempts if a["accepted"])
        flipped_count = sum(1 for a in attempts if a.get("flipped", False))

        # Additional statistics based on cand_type
        cand_types = set(attempt.get('cand_type') for attempt in attempts_data if attempt.get('cand_type'))
        cand_type_stats = {}
        for ct in cand_types:
            ct_total = sum(1 for a in attempts_data if a.get('cand_type') == ct)
            ct_accepted = sum(1 for a in attempts_data if a.get('cand_type') == ct and a["accepted"])
            ct_flipped = sum(1 for a in attempts_data if a.get('cand_type') == ct and a["flipped"])

            cand_type_stats[ct] = {
                "total": ct_total,
                "flipped": ct_flipped,
                "accepted": ct_accepted,
                "acceptance_rate": ct_accepted / ct_total if ct_total > 0 else 0,
                "flip_rate": ct_flipped / ct_total if ct_total > 0 else 0
            }

        print(f"\n{'='*60}")
        print(f"PROBING RESULTS")
        print(f"{'='*60}")
        print(f"Total proposals:        {total_proposals}")
        print(f"Generator rejections:   {generator_rejections} ({generator_rejections/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"Model flipped:          {flipped_count} ({flipped_count/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"NLI accepted:           {accepted_count} ({accepted_count/total_proposals*100:.1f}%)" if total_proposals > 0 else "")
        print(f"Overall ASR:            {flipped_count/total_proposals:.3f}" if total_proposals > 0 else "")
        if "duration_seconds" in locals() or "duration" in locals():
            d = locals().get("duration") or locals().get("duration_seconds")
            print(f"Time Taken:             {d:.1f}s")
        print(f"{'=' * 60}\n")

        print("\nCand Type Statistics:")
        for ct, stats in cand_type_stats.items():
            print(f"  {ct}:")
            print(f"    Total: {stats['total']}")
            print(f"    Accepted: {stats['accepted']} ({stats['acceptance_rate']:.2%})")
            print(f"    Flipped: {stats['flipped']} ({stats['flip_rate']:.2%})")

        print(f"{'='*60}\n")

        generator_rejection_rate = generator_rejections / total_proposals if total_proposals > 0 else 0
        acceptance_rate = accepted_count / total_proposals if total_proposals > 0 else 0
        flip_on_accepted_rate = flipped_count / accepted_count if accepted_count > 0 else 0
        overall_success_rate = flipped_count / total_proposals if total_proposals > 0 else 0
        
        # NLI score distributions
        valid_attempts = [a for a in attempts if "generator_rejected" not in a and a["nli_scores"]]
        if valid_attempts:
            forward_scores = [a["nli_scores"]["forward"] for a in valid_attempts]
            backward_scores = [a["nli_scores"]["backward"] for a in valid_attempts]
            
            nli_stats = {
                "forward": {
                    "mean": np.mean(forward_scores),
                    "median": np.median(forward_scores),
                    "q25": np.percentile(forward_scores, 25),
                    "q75": np.percentile(forward_scores, 75)
                },
                "backward": {
                    "mean": np.mean(backward_scores),
                    "median": np.median(backward_scores),
                    "q25": np.percentile(backward_scores, 25),
                    "q75": np.percentile(backward_scores, 75)
                }
            }
        else:
            nli_stats = None
        
        return {
            "probe_config": {
                "task": self.task,
                "model": self.model_key,
                "explainer": self.explainer,
                "strategy": self.strategy,
                "threshold": self.threshold
            },
            "results": {
                "total_proposals": total_proposals,
                "generator_rejections": generator_rejections,
                "generator_rejection_rate": generator_rejection_rate,
                "accepted_count": accepted_count,
                "acceptance_rate": acceptance_rate,
                "flipped_count": flipped_count,
                "flip_on_accepted_rate": flip_on_accepted_rate,
                "overall_success_rate": overall_success_rate,
                "nli_score_distributions": nli_stats
            },
            "files": {
                "attempts": str(attempts_file),
                "accepted": str(failures_file)
            }
        }
    

    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        if self.is_already_run():
            print(f"Skipping: Experiment already exists in master_results.csv for {self.task}/{self.model}/{self.strategy}...")
            self.logger.info("Experiment skip requested: matched record found in master table.")
            return None
        print(f"\n{'='*60}")
        print(f"STARTING LexCheck EXPERIMENT")
        print(f"{'='*60}")
        print(f"Task:        {self.task}")
        print(f"Model:       {self.model}")
        print(f"Explainer:   {self.explainer}")
        print(f"Strategy:    {self.strategy}")
        print(f"Threshold:   {self.threshold}")
        print(f"Experiment:  {self.experiment_dir.name}")
        print(f"{'='*60}\n")

        self.logger.info("Starting full experiment")
        self.logger.info(f"Task: {self.task}, Model: {self.model}")
        self.logger.info(f"Explainer: {self.explainer}, Strategy: {self.strategy}, Threshold: {self.threshold}")
        
        try:
            # Run pipeline
            mining_result = self.mine_errors()
            probe_result = self.probe_edits()
            # aft_result = self.train_aft()
            
            # Save summary
            summary = {
                "experiment_config": {
                    "task": self.task,
                    "model": self.model,
                    "explainer": self.explainer,
                    "strategy": self.strategy,
                    "threshold": self.threshold
                },
                "results": {
                    "mining": mining_result,
                    "probe": probe_result,
                    #"aft": aft_result
                },
                "completed_at": datetime.now().isoformat()
            }
            
            with open(self.experiment_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n{'='*60}")
            print(f"✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Results saved to: {self.experiment_dir}")
            print(f"{'='*60}\n")

            self.logger.info("Experiment completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Run focused LexCheck experiment")
    
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., sst2)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., distilbert-base-uncased)")
    parser.add_argument("--explainer", type=str, required=True,
                       choices=["ig", "attn", "occlusion","random"],
                       help="Explainer type")
    parser.add_argument("--strategy", type=str, required=True,
                       choices=["lm", "random", "prefix"],
                       help="Placement strategy")
    parser.add_argument("--threshold", type=float, required=True,
                       help="NLI threshold")
    
    parser.add_argument("--steps", type=str,
                       help="Comma-separated steps to run (mine,probe,aft)")
    parser.add_argument("--foundation-dir", type=Path, default=Path("foundation"),
                       help="Foundation directory path")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"),
                       help="Experiments directory path")
    parser.add_argument("--experiment-name", type=str,
                       help="Custom experiment name")
    parser.add_argument("--max-probe-items", type=int,
                       help="Maximum number of items to probe (default: all)")
    parser.add_argument("--candidates-per-item", type=int, default=1,
                       help="Number of candidates to test per item (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for NLI and flip checks (default: 64, higher = faster)")
    parser.add_argument("--operation", type=str, choices=["inject", "ablate"], default="inject",
                        help="Operation to perform on generated edits (default: inject)")

    debug_input = [
        "--task", "github",
        "--model", "Qwen/Qwen2.5-0.5B_gen",#"distilbert-base-uncased",#"Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B_gen","facebook/bart-large"
        "--explainer", "ig",#"ig","occlusion","attn","random"
        "--strategy", "prefix", #"lm", "random", "prefix"
        "--threshold", "0.0",
        "--operation", "inject", # "ablate"
        "--steps", "mine,probe",
        #"--max-probe-items", 10
    ]

    args = parser.parse_args() # debug_input

    #try:
    runner = ExperimentRunner(
        task=args.task,
        model=args.model,
        explainer=args.explainer,
        strategy=args.strategy,
        threshold=args.threshold,
        foundation_dir=args.foundation_dir,
        experiments_dir=args.experiments_dir,
        experiment_name=args.experiment_name,
        max_probe_items=args.max_probe_items,
        candidates_per_item=args.candidates_per_item,
        batch_size=args.batch_size,
        operation=args.operation
    )

    if args.steps:

        if runner.is_already_run():
            print(f"⚠ Warning: Configuration exists in master_results.csv. ")
            return None
        steps = [s.strip() for s in args.steps.split(",")]
        for step in steps:
            if step == "mine":
                runner.mine_errors()
            elif step == "probe":
                runner.probe_edits()
            #elif step == "aft":
            #    runner.train_aft()
            else:
                print(f"Unknown step: {step}")
    else:
        # Run full experiment
        runner.run_full_experiment()

    print(f"✓ Experiment completed: {runner.experiment_dir}")

    #except Exception as e:
    #    print(f"✗ Experiment failed: {e}")
    #    sys.exit(1)

if __name__ == "__main__":
    main()
