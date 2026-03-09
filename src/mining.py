import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from captum.attr import IntegratedGradients, Occlusion, LayerIntegratedGradients
from tqdm import tqdm
import math
import time
import spacy
import spacy.cli
import random
from datetime import datetime
from .config import ExperimentConfig
from .data_utils import load_split_data, prepare_model_inputs, get_task_prompt, get_label_names
from .utils import save_jsonl, load_jsonl, save_partial_results, load_partial_results

def get_attention_rollout(model, inputs, head_fusion="mean", target_token_index=None, discard_first_token=None):
    # 1. Force output_attentions
    original_config_setting = getattr(model.config, "output_attentions", None)
    model.config.output_attentions = True

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    if original_config_setting is not None:
        model.config.output_attentions = original_config_setting

    if attentions is None:
        raise ValueError("Model returned None. Use `attn_implementation='eager'`.")

    # 2. Auto-detect model type
    is_decoder = getattr(model.config, "is_decoder", False)
    if not is_decoder and getattr(model.config, "model_type", "") in ["qwen2", "llama", "mistral"]:
        is_decoder = True

    if target_token_index is None:
        target_token_index = -1 if is_decoder else 0

    if discard_first_token is None:
        discard_first_token = is_decoder

    num_layers = len(attentions)
    batch_size, num_heads, seq_len, _ = attentions[0].shape
    device = attentions[0].device

    # 3. Head Fusion
    if head_fusion == "mean":
        attentions = [attn.mean(dim=1) for attn in attentions]
    elif head_fusion == "max":
        attentions = [attn.max(dim=1)[0] for attn in attentions]

    # 4. Dynamically adjust the attention ratio based on the model type
    # BERT (Encoder, ~12 layers): 0.5
    # LLM (Decoder, ~24+ layers): 0.1
    residual_ratio = 0.1 if is_decoder else 0.5
    attn_ratio = 1.0 - residual_ratio

    identity = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    # Pre-process all layers: Mix and Normalize immediately
    processed_attentions = []
    for attn in attentions:
        mixed = attn_ratio * attn + residual_ratio * identity
        # Row-normalize immediately to keep values stable
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        processed_attentions.append(mixed)

    # 5. Rollout Calculation (Fast GPU Matrix Multiplication)
    rollout = processed_attentions[0]
    for i in range(1, num_layers):
        rollout = torch.bmm(processed_attentions[i], rollout)

    # 6. Extract target token row
    cls_attention = rollout[:, target_token_index, :]

    # 7. Post-processing (Discard First Token & Renormalize)
    if discard_first_token and seq_len > 1:
        # Set first token attention to 0
        cls_attention[:, 0] = 0.0

        # Renormalize the REST to sum to 1
        sum_val = cls_attention.sum(dim=-1, keepdim=True)

        # Avoid division by zero if everything was 0
        cls_attention = cls_attention / (sum_val + 1e-9)

    # Double check normalization (sometimes float precision messes it up)
    final_scores = cls_attention.cpu().numpy()

    return final_scores # .tolist()


def get_grad_weighted_attention_bart(model, inputs, target_class):
    model.zero_grad()
    extracted_grads = []
    extracted_attns = []

    def hook_fn(module, input, output):
        attn_weights = output[0]
        attn_weights.retain_grad()
        extracted_attns.append(attn_weights)

    # Target layer path remains unchanged
    if hasattr(model, "model"):
        target_layer = model.model.decoder.layers[-1].encoder_attn
    else:
        target_layer = model.decoder.layers[-1].encoder_attn

    handle = target_layer.register_forward_hook(hook_fn)

    outputs = model(**inputs)
    logits = outputs.logits

    # If target_class is a scalar, convert to tensor for backward pass
    if isinstance(target_class, int):
        score = logits[0, target_class]
    else:
        score = logits[0, target_class[0]] if hasattr(target_class, "__len__") else logits[0, target_class]

    score.backward()
    handle.remove()

    if not extracted_attns:
        return np.zeros((inputs["input_ids"].shape[0], inputs["input_ids"].shape[1]))

    attn = extracted_attns[0]
    grad = attn.grad

    # Grad-weighted attention
    # [Batch, Heads, Seq_Out, Seq_In] -> [Batch, Seq_Out, Seq_In]
    weighted_attn = (grad * attn).clamp(min=0).mean(dim=1)

    # --- Fix core: Adaptive dimension check ---
    if weighted_attn.dim() == 3:
        # If 3D [Batch, Seq_Out, Seq_In], take the last output token attention
        final_scores = weighted_attn[:, -1, :].detach().cpu().numpy()
    elif weighted_attn.dim() == 2:
        # If already 2D [Batch, Seq_In], convert directly to numpy
        final_scores = weighted_attn.detach().cpu().numpy()
    else:
        # Fallback: If dimension is abnormal, try flattening
        final_scores = weighted_attn.view(weighted_attn.size(0), -1).detach().cpu().numpy()

    return final_scores

def get_random_scores(inputs: Dict[str, torch.Tensor]) -> List[float]:
    """
    Generates random attribution scores for each token as a baseline.
    Scores are drawn from a uniform distribution [0, 1].
    """
    seq_len = inputs["input_ids"].shape[1]
    # Generate random scores for the sequence
    scores = [random.random() for _ in range(seq_len)]
    return scores


import torch
from captum.attr import LayerIntegratedGradients


def get_integrated_gradients_bart(model, inputs, target_class):
    # 1. Prepare input data
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", None)

    # --- Key fix step 1: Construct a valid Baseline ---
    # Captum's default all-zero baseline causes BART to crash when unable to find EOS
    # We create a baseline: keep EOS token, replace everything else with PAD token

    # Get special token IDs
    eos_token_id = model.config.eos_token_id
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0

    # Clone input_ids as baseline
    baseline_input_ids = input_ids.clone()
    # Replace all non-EOS positions with PAD
    # Note: This assumes the input always contains EOS. If not, BART itself cannot handle it.
    mask_not_eos = (baseline_input_ids != eos_token_id)
    baseline_input_ids[mask_not_eos] = pad_token_id

    # ------------------------------------------

    # 2. Define Forward function
    def forward_func(input_ids, attention_mask=None, token_type_ids=None):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = model(**model_inputs)
        logits = outputs.logits
        if len(logits.shape) == 3:
            return logits[:, -1, :]
        return logits

    # 3. Get the correct Embedding layer (handle DataParallel and BART structure)
    if hasattr(model, "module"):
        model_to_use = model.module
    else:
        model_to_use = model

    try:
        # Try to directly get Encoder's Embeddings (BART/Marian etc.)
        embedding_layer = model_to_use.model.encoder.embed_tokens
    except AttributeError:
        # Fallback (BERT etc.)
        embedding_layer = model_to_use.get_input_embeddings()

    # 4. Initialize LayerIG
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    # 5. Prepare additional arguments
    additional_args = (attention_mask, token_type_ids)

    # 6. Calculate attribution (pass baselines)
    # Note: Must pass baselines=baseline_input_ids here
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,  # <--- Core fix
        additional_forward_args=additional_args,
        target=target_class,
        n_steps=50,
        internal_batch_size=16
    )

    # 7. Aggregate results
    token_attributions = attributions.sum(dim=-1).squeeze(0)
    scores = token_attributions.detach().cpu().numpy()

    if scores.ndim == 0:
        scores = [float(scores)]
    else:
        scores = scores.tolist()

    return scores

def get_integrated_gradients(model, inputs, target_class):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # batch_size = input_ids.shape[0]

    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    embeddings.requires_grad_(True)

    def forward_func(inputs_embeds):
        # inputs_embeds size: [batch_size * n_steps, seq_len, hidden_dim]
        # We need to expand attention_mask to match the attribution batching

        current_batch_size = inputs_embeds.shape[0]
        # steps = curr_total_batch // batch_size

        expanded_mask = attention_mask.expand(current_batch_size, -1)

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": expanded_mask
        }

        if "token_type_ids" in inputs:
            expanded_token_types = inputs["token_type_ids"].expand(current_batch_size, -1)
            model_inputs["token_type_ids"] = expanded_token_types

        logits = model(**model_inputs).logits
        # if it is a causal model, 3d output [batch_size, seq_len, vocab_size]
        if len(logits.shape) == 3:
            return logits[:, -1, :]
        return logits
    
    ig = IntegratedGradients(forward_func)

    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(embeddings, baseline, target=target_class, n_steps=50, internal_batch_size=16)

    token_attributions = attributions.sum(dim=-1).squeeze(0)
    scores = token_attributions.detach().cpu().numpy()

    if scores.ndim == 0:
        scores = [float(scores)]
    else:
        scores = scores.tolist()

    return scores


def get_occlusion_bart(model, inputs, target_class, window_len=1):
    """
    Occlusion attribution optimized for BART.
    Fixes two critical issues:
    1. Batch dimension alignment issue (solved by processing per sample).
    2. EOS Token loss causing model crash (solved by forced recovery of EOS).
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]

    # Results container
    all_scores = []

    # Get special Token IDs
    if model.config.pad_token_id is not None:
        pad_id = model.config.pad_token_id
    else:
        pad_id = 0

    eos_token_id = model.config.eos_token_id  # BART must have EOS

    # -------------------------------------------------------
    # Process each sample individually (Loop over batch)
    # -------------------------------------------------------
    for i in range(batch_size):
        # 1. Extract single sample [1, seq_len]
        curr_input_ids = input_ids[i:i + 1]
        curr_attention_mask = attention_mask[i:i + 1]

        # --- Key step: Find EOS positions in current sample ---
        # BART may have multiple EOS tokens (e.g., mid-sentence and end), usually protect all EOS
        # nonzero returns indices of (batch_idx, seq_idx), we only need seq_idx
        eos_indices = (curr_input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
        # ---------------------------------------

        # Handle token_type_ids
        curr_token_type_ids = None
        if "token_type_ids" in inputs:
            curr_token_type_ids = inputs["token_type_ids"][i:i + 1]

        # 2. Define Forward function
        def forward_func(perturbed_input_ids, mask_ref=None, token_type_ref=None):
            # perturbed_input_ids: [k, seq_len] (k = num_perturbations)

            # --- Core fix: Forcefully restore EOS Token ---
            # No matter how Captum occludes, we always write EOS back
            # This ensures BART always sees the same number of EOS and won't error
            for eos_idx in eos_indices:
                perturbed_input_ids[:, eos_idx] = eos_token_id
            # ----------------------------------

            current_k = perturbed_input_ids.shape[0]

            # Expand Mask
            expanded_mask = mask_ref.expand(current_k, -1)

            model_inputs = {
                "input_ids": perturbed_input_ids,
                "attention_mask": expanded_mask
            }

            if token_type_ref is not None:
                model_inputs["token_type_ids"] = token_type_ref.expand(current_k, -1)

            outputs = model(**model_inputs)
            logits = outputs.logits

            if len(logits.shape) == 3:
                return logits[:, -1, :]
            return logits

        # 3. Initialize Captum
        ablator = Occlusion(forward_func)

        sliding_window_shapes = (window_len,)
        strides = (1,)
        additional_args = (curr_attention_mask, curr_token_type_ids)

        curr_target = target_class[i] if isinstance(target_class, (list, np.ndarray, torch.Tensor)) else target_class

        # 4. Execute attribution
        try:
            attributions = ablator.attribute(
                inputs=curr_input_ids,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                baselines=pad_id,
                target=curr_target,
                additional_forward_args=additional_args,
                perturbations_per_eval=16  # Can increase if GPU memory permits, e.g., 64
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                attributions = ablator.attribute(
                    inputs=curr_input_ids,
                    sliding_window_shapes=sliding_window_shapes,
                    strides=strides,
                    baselines=pad_id,
                    target=curr_target,
                    additional_forward_args=additional_args,
                    perturbations_per_eval=1  # Extreme case
                )
            else:
                raise e

        # 5. Collect results
        score = attributions.squeeze(0).detach().cpu().numpy().tolist()
        all_scores.append(score)

    return all_scores

def get_occlusion(model, inputs, target_class, window_len=1):
    """
    Uses Captum's Occlusion to get the attribution scores for each token
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]

    # Obtain Embedding
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)

    # hidden_size (e.g. 768 or 1024)
    hidden_dim = embeddings.shape[-1]

    def forward_func(inputs_embeds):
        current_batch_size = inputs_embeds.shape[0]
        expanded_mask = attention_mask.repeat_interleave(current_batch_size // batch_size, dim=0)

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": expanded_mask
        }

        if "token_type_ids" in inputs:
            model_inputs["token_type_ids"] = inputs["token_type_ids"].expand(current_batch_size, -1)
        logits = model(**model_inputs).logits

        # if it is a causal model, 3d output [batch_size, seq_len, vocab_size]
        if len(logits.shape) == 3:
            return logits[:, -1, :]
        return logits

    ablator = Occlusion(forward_func)

    sliding_window_shapes = (window_len, hidden_dim)

    strides = (1, hidden_dim)

    try:
        attributions = ablator.attribute(
            inputs=embeddings,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=0,  # use zero to occlude
            target=target_class,
            perturbations_per_eval=4 #32
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM during Occlusion, retrying with smaller batch...")
            torch.cuda.empty_cache()
            attributions = ablator.attribute(
                inputs=embeddings,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                baselines=0,
                target=target_class,
                perturbations_per_eval=1
            )
        else:
            raise e

    token_attributions = attributions.sum(dim=-1).squeeze(0)
    scores = token_attributions.detach().cpu().numpy()

    if scores.ndim == 0:
        scores = [float(scores)]
    else:
        scores = scores.tolist()

    return scores

def mine_errors_for_task(config: ExperimentConfig, run_dir: Path, task: str,
                        model_key: str) -> Dict[str, Any]:
    model_path = run_dir / "models" / task / model_key

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, fix_mistral_regex=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if getattr(config, "use_generative_mode") else torch.float32
    # Use eager attention implementation to support output_attentions=True
    # This fixes the "sdpa" error for Qwen/Llama models
    if getattr(config, "use_generative_mode"):
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            attn_implementation="eager",
            dtype=dtype
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            attn_implementation="eager",
            dtype=dtype
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    mining_items = load_split_data(run_dir, task, "mining")

    if hasattr(config, "max_items") and config.max_items is not None:
        mining_items = mining_items[:config.max_items]
        print(f"Truncated mining items to {len(mining_items)} due to max_items config.")
    # ACCELERATION: Batch size for inference
    BATCH_SIZE = getattr(config, "batch_size", 16)
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    misclassified_items, correctly_classified_items = [], []

    # Initialize timing statistics
    timing_stats = {signal: {"count": 0, "total_time": 0.0} for signal in config.signals}

    # Process in Batches instead of individual items
    num_batches = math.ceil(len(mining_items) / BATCH_SIZE)
    #
    """partial_file = run_dir / "pools" / task / f"mining_partial_{model_key}.pkl"
    partial_data = load_partial_results(partial_file)
    
    if partial_data:
        start_idx = partial_data["last_processed"] + 1
        misclassified_items = partial_data["misclassified_items"]
        correctly_classified_items = partial_data.get("correctly_classified_items", [])
        print(f"Resuming from item {start_idx}")
    else:
        start_idx = 0
        misclassified_items = []
        correctly_classified_items = []"""

    for batch_idx in tqdm(range(num_batches), desc=f"Mining {task} {model_key}"):
        batch_items = mining_items[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]

        if getattr(config, "use_generative_mode", False):
            # Generative logic
            label_names = get_label_names(task)
            prompts = [get_task_prompt(task, item["text_a"], item.get("text_b")) for item in batch_items]
            encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **encodings,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    #eos_token_id=tokenizer.eos_token_id
                )
            responses = tokenizer.batch_decode(generated_ids[:, encodings.input_ids.shape[1]:], skip_special_tokens=True)

            indexed_labels = sorted(enumerate(label_names), key=lambda x: len(x[1]), reverse=True)
            predicted_labels = []
            for response in responses:
                res_lower = response.lower().strip()

                # --- Use find() to locate the earliest occurring label ---
                pred_idx = -1
                earliest_pos = float('inf')

                for label_id, name in enumerate(label_names):
                    pos = res_lower.find(name.lower())

                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                        pred_idx = label_id

                # Fallback: if no valid label found, default to 0 (avoid -1 causing evaluation metric errors)
                if pred_idx == -1:
                    pred_idx = 0
                predicted_labels.append(pred_idx)
                # true_label = item["label"]
        else:
            encodings, _ = prepare_model_inputs(batch_items, tokenizer, task=task, is_generative=False)
            encodings = {k: v.long().to(device) for k, v in encodings.items()}
            # for discriminative models
            with torch.no_grad():
                outputs = model(**encodings)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1).cpu().numpy().tolist()

        # Phase 2: Generate explanations for ALL items in batch
        batch_explanations = {signal: [] for signal in config.signals}

        for idx in range(len(batch_items)):
            item_encodings = {k: v[idx:idx + 1] for k, v in encodings.items()}
            pred_label = predicted_labels[idx]
            item = batch_items[idx]

            # Determine attribution target
            attribution_target = pred_label
            if getattr(config, "use_generative_mode", False) and pred_label != -1:
                label_name = label_names[pred_label]
                target_token_ids = tokenizer.encode(label_name, add_special_tokens=False)
                attribution_target = target_token_ids[0] if len(target_token_ids) > 0 else (tokenizer.eos_token_id or 0)

            # Skip XAI if prediction failed (generative -1)
            if pred_label == -1:
                for sig in config.signals:
                    batch_explanations[sig].append([0.0] * item_encodings['input_ids'].shape[1])
                continue

            # Calculate Raw Signals
            current_item_explanations = {}

            if "ig" in config.signals:
                start_time = time.time()

                try:
                    if "bart" in model.config.model_type.lower():
                        current_item_explanations["ig"] = get_integrated_gradients_bart(model, item_encodings,
                                                                                        attribution_target)
                    else:
                        current_item_explanations["ig"] = get_integrated_gradients(model, item_encodings,
                                                                                       attribution_target)
                except Exception as e:
                    print(f"IG failed: {e}")
                    current_item_explanations["ig"] = [0.0] * item_encodings['input_ids'].shape[1]
                elapsed_time = time.time() - start_time
                timing_stats["ig"]["count"] += 1
                timing_stats["ig"]["total_time"] += elapsed_time

            if "attn" in config.signals:
                start_time = time.time()
                try:
                    if "bart" in model.config.model_type.lower():
                        # Use gradient-weighted scheme for BART
                        attn_res = get_grad_weighted_attention_bart(model, item_encodings, attribution_target)
                        current_item_explanations["attn"] = attn_res[0].tolist()
                    elif "bert" in model.config.model_type.lower():
                        # Keep original Rollout scheme for other models
                        attn_res = get_attention_rollout(model, item_encodings)
                        current_item_explanations["attn"] = attn_res[0].tolist()
                    else:
                        print("Attention only supported for BART and BERT models")
                except Exception as e:
                    print(f"Attention failed: {e}")
                    current_item_explanations["attn"] = [0.0] * item_encodings['input_ids'].shape[1]
                elapsed_time = time.time() - start_time
                timing_stats["attn"]["count"] += 1
                timing_stats["attn"]["total_time"] += elapsed_time

            if "occlusion" in config.signals:
                start_time = time.time()
                try:
                    if "bart" in model.config.model_type.lower():
                        bart_occlusion_res = get_occlusion_bart(model, item_encodings, attribution_target)
                        current_item_explanations["occlusion"] = bart_occlusion_res[0]
                        # print(f"current occlusion result: {current_item_explanations['occlusion']}")
                    else:
                        current_item_explanations["occlusion"] = get_occlusion(model, item_encodings, attribution_target)
                except Exception as e:
                    print(f"Occlusion failed: {e}")
                    current_item_explanations["occlusion"] = [0.0] * item_encodings['input_ids'].shape[1]
                elapsed_time = time.time() - start_time
                timing_stats["occlusion"]["count"] += 1
                timing_stats["occlusion"]["total_time"] += elapsed_time

            if "random" in config.signals:
                start_time = time.time()
                current_item_explanations["random"] = get_random_scores(item_encodings)
                elapsed_time = time.time() - start_time
                timing_stats["random"]["count"] += 1
                timing_stats["random"]["total_time"] += elapsed_time

            # NEW: Filter Prompt/Template tokens for ALL models (especially generative)
            tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][idx])
            # raw text content for excluding template tokens and symbols
            raw_text_content = (item["text_a"] + " " + (item.get("text_b") or "")).lower()

            for sig in config.signals:
                if sig in current_item_explanations:
                    scores = current_item_explanations[sig]
                    filtered_scores = []

                    # Fix IndexError: ensure scores and tokens have the same length
                    if len(scores) != len(tokens):
                        # If lengths differ, log warning and perform simple alignment
                        # Usually take whichever is shorter, or handle based on model characteristics
                        min_len = min(len(scores), len(tokens))
                        # If scores too short, pad with 0; if scores too long, truncate
                        if len(scores) < len(tokens):
                            scores = list(scores) + [0.0] * (len(tokens) - len(scores))
                        else:
                            scores = scores[:len(tokens)]

                    for t_idx, t in enumerate(tokens):
                        clean_t = t.replace('Ġ', '').replace('##', '').replace(' ', '').lower()
                        # filter conditions:
                        # 1. Token must be part of original text
                        # 2. Exclude common template prompts
                        # 3. Exclude special characters
                        is_valid = (clean_t and clean_t in raw_text_content
                                    # and clean_t not in ["sentiment:", "category:", "instruction:", "input:","context:", "question:"]
                                    )
                        # print(f"DEBUG {clean_t} is_valid: {is_valid} in \n {raw_text_content} ")
                        filtered_scores.append(scores[t_idx] if is_valid else 0.0)
                    current_item_explanations[sig] = filtered_scores

            for sig in config.signals:
                batch_explanations[sig].append(current_item_explanations.get(sig, []))

        # Phase 3: Separate and Save
        for idx, item in enumerate(batch_items):
            true_label = item["label"]
            pred_label = predicted_labels[idx]
            tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][idx])

            item_explanations = {sig: batch_explanations[sig][idx] for sig in config.signals}

            result_data = {
                "id": item["id"],
                "text_a": item["text_a"],
                "text_b": item.get("text_b"),
                "true_label": true_label,
                "predicted_label": pred_label,
                "tokens": tokens,
                "explanations": item_explanations
            }

            if pred_label != true_label:
                misclassified_items.append(result_data)
            else:
                correctly_classified_items.append(result_data)

        """if i % 100 == 99:
            save_partial_results(partial_file, {
                "last_processed": i,
                "misclassified_items": misclassified_items,
                "correctly_classified_items": correctly_classified_items
            })"""

    misclassified_file = run_dir / "pools" / task / f"mined_errors_{model_key}.jsonl"
    correctly_classified_file = run_dir / "pools" / task / f"correctly_classified_{model_key}.jsonl"
    misclassified_file.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(misclassified_file, misclassified_items)
    save_jsonl(correctly_classified_file, correctly_classified_items)

    """if partial_file.exists():
        partial_file.unlink()"""

    # Calculate average timing for each XAI method
    timing_summary = {}
    for signal in config.signals:
        count = timing_stats[signal]["count"]
        total_time = timing_stats[signal]["total_time"]
        timing_summary[signal] = {
            "count": count,
            "total_time": total_time,
            "average_time": total_time / count if count > 0 else 0.0
        }

    # Save timing statistics to JSON file
    timing_file = run_dir / "pools" / task / f"xai_timing_summary.json"

    # 1. Load existing stats if file exists, otherwise start fresh
    all_timing_data = {}
    if timing_file.exists():
        try:
            with open(timing_file, "r") as f:
                all_timing_data = json.load(f)
        except Exception:
            all_timing_data = {}

    # 2. Add current model results and timestamp
    all_timing_data[model_key] = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals": timing_summary
    }

    # 3. Overwrite the file with the merged data
    with open(timing_file, "w") as f:
        json.dump(all_timing_data, f, indent=2)

    print(f"XAI timing statistics saved to: {timing_file}")

    return {
        "num_misclassified": len(misclassified_items),
        "num_correctly_classified": len(correctly_classified_items),
        "total_mining_items": len(mining_items),
        "error_rate": len(misclassified_items) / len(mining_items),
        "accuracy": len(correctly_classified_items) / len(mining_items),
        "misclassified_file": str(misclassified_file),
        "correctly_classified_file": str(correctly_classified_file),
        "xai_timing_stats": timing_summary
    }


def tokens_to_words(tokens: List[str], scores: List[float], model_key: str) -> List[Tuple[str, float, List[int]]]:
    words = []
    current_word = ""
    current_score = 0.0
    current_indices = []

    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "[UNK]", "<|endoftext|>", "<|file_separator|>",
                      "<|extra_0|>"}

    punctuation = {'.', ',', '?', '!', ';', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"',
                   '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`'}

    for i, (token, score) in enumerate(zip(tokens, scores)):
        # 1. Filter unnecessary tokens
        if token in special_tokens or token in punctuation or not token.strip():
            if current_word:
                words.append((current_word, current_score, current_indices))
                current_word, current_score, current_indices = "", 0.0, []
            continue

        # Handle prefix due to the tokenizer
        # BERT style ## as continuation,
        # Qwen/Llama: If it does not begin with Ġ or " " and is not the first token, then it is a continuation.
        if any(name in model_key.lower() for name in ["bert", "roberta", "albert", "electra"]):
            is_continuation = token.startswith("##")
        else:
            has_leading_space = token.startswith("Ġ") or token.startswith(" ")
            is_continuation = (not has_leading_space and
                            i > 0 and
                            tokens[i-1] not in punctuation)

        #print(f"debug {token} is_continuation: {is_continuation}")
        clean_token = token.lstrip("##").replace('Ġ', '').strip()

        if is_continuation:
            current_word += clean_token
            current_score += score  # directly add raw score, keep + or -
            current_indices.append(i)
        else:
            if current_word and current_word.strip() not in punctuation:
                words.append((current_word, current_score, current_indices))
            current_word = clean_token
            current_score = score
            current_indices = [i]
        #print(f"debug: {current_word} {current_score} {current_indices} is_continuation: {is_continuation}")

    if current_word and current_word.strip() not in punctuation:
        words.append((current_word, current_score, current_indices))

    return words

def build_candidate_pools(config: ExperimentConfig, run_dir: Path, task: str,
                         model_key: str) -> Dict[str, Any]:
    misclassified_file = run_dir / "pools" / task / f"mined_errors_{model_key}.jsonl"
    correct_file = run_dir / "pools" / task / f"correctly_classified_{model_key}.jsonl"

    misclassified_items = load_jsonl(misclassified_file) if misclassified_file.exists() else []
    correct_items = load_jsonl(correct_file) if correct_file.exists() else []

    pools_info = {}

    for signal in config.signals:

        for items_list, filename_prefix in [(misclassified_items, "pool"), (correct_items, "correct_pool")]:
            all_candidates = []

            for item in items_list:
                if signal not in item["explanations"]: continue
                words = tokens_to_words(item["tokens"], item["explanations"][signal], model_key=model_key)
                if not words: continue

                # 1. Extract most positive word
                top_pos = max(words, key=lambda x: x[1])
                if top_pos[1] > 0:
                    all_candidates.append({
                        "item_id": item["id"], "word": top_pos[0], "score": top_pos[1], "type": "max_pos"
                    })

                # 2. Extract most negative word
                top_neg = min(words, key=lambda x: x[1])
                if top_neg[1] < 0:
                    all_candidates.append({
                        "item_id": item["id"], "word": top_neg[0], "score": top_neg[1], "type": "max_neg"
                    })

            pool_file = run_dir / "pools" / task / f"{filename_prefix}_{signal}_{model_key}.jsonl"
            save_jsonl(pool_file, all_candidates)
            pools_info[f"{filename_prefix}_{signal}"] = len(all_candidates)

    return pools_info


def compute_pool_overlap(config: ExperimentConfig, run_dir: Path, task: str,
                        model_key: str) -> Dict[str, Any]:
    if not config.signals:
        return {}

    error_pools_data = {}
    correct_pools_data = {}

    # 1. Load all generated pool data
    for signal in config.signals:
        e_file = run_dir / "pools" / task / f"pool_{signal}_{model_key}.jsonl"
        if e_file.exists():
            error_pools_data[signal] = set(item.get("word", "") for item in load_jsonl(e_file))

        c_file = run_dir / "pools" / task / f"correct_pool_{signal}_{model_key}.jsonl"
        if c_file.exists():
            correct_pools_data[signal] = set(item.get("word", "") for item in load_jsonl(c_file))

    overlaps = {
        "cross_signal_error": {},  # Consensus across different XAI signals on error samples
        "cross_signal_correct": {},  # Consensus across different XAI signals on correct samples
        "error_vs_correct": {}  # Feature overlap between correct/incorrect samples for same signal
    }

    signals = list(config.signals)

    # 2. Calculate overlaps
    for i in range(len(signals)):
        sig_i = signals[i]

        # Dimension A: Error vs Correct (same signal)
        if sig_i in error_pools_data and sig_i in correct_pools_data:
            s_e, s_c = error_pools_data[sig_i], correct_pools_data[sig_i]
            inter = s_e & s_c
            uni = s_e | s_c
            overlaps["error_vs_correct"][sig_i] = {
                "jaccard": len(inter) / len(uni) if uni else 0,
                "intersection_size": len(inter)
            }

        # Dimension B: Cross-signal (different signals)
        for j in range(i + 1, len(signals)):
            sig_j = signals[j]
            pair_key = f"{sig_i}_vs_{sig_j}"

            # Cross-signal comparison on error pools
            if sig_i in error_pools_data and sig_j in error_pools_data:
                s_a, s_b = error_pools_data[sig_i], error_pools_data[sig_j]
                inter, uni = s_a & s_b, s_a | s_b
                overlaps["cross_signal_error"][pair_key] = {
                    "jaccard": len(inter) / len(uni) if uni else 0,
                    "intersection_size": len(inter)
                }

            # Cross-signal comparison on correct pools
            if sig_i in correct_pools_data and sig_j in correct_pools_data:
                s_a, s_b = correct_pools_data[sig_i], correct_pools_data[sig_j]
                inter, uni = s_a & s_b, s_a | s_b
                overlaps["cross_signal_correct"][pair_key] = {
                    "jaccard": len(inter) / len(uni) if uni else 0,
                    "intersection_size": len(inter)
                }

    return overlaps

def analyze_pos_tags(nlp_model, config: ExperimentConfig, run_dir: Path, task: str, model_key: str) -> Dict[str, Any]:
    """
    Loads collected pools, performs POS analysis on words, and records POS tags/labels in a JSONL document.

    Args:
        nlp_model: The loaded spaCy language model.
        config: Experiment configuration object.
        run_dir: The base directory for the run.
        task: The name of the task (e.g., 'sst2').
        model_key: The key identifying the model (e.g., 'distilbert_base_uncased').

    Returns:
        A dictionary containing POS analysis results for each signal.
    """
    pos_analysis_results = {}
    pools_base_path = run_dir / "pools" / task

    for signal in config.signals:
        pool_file = pools_base_path / f"pool_{signal}_{model_key}.jsonl"
        if not pool_file.exists():
            print(f"Warning: Pool file not found for signal '{signal}' and model '{model_key}': {pool_file}")
            continue

        print(f"Analyzing POS tags for pool: {pool_file}")
        pool_data = load_jsonl(pool_file)

        analyzed_words = []
        pos_counts = Counter()
        tag_counts = Counter()
        label_counts = Counter()
        for item in pool_data:
            word = item.get("word", "").strip()
            if not word:
                continue

            doc = nlp_model(word)
            for token in doc:
                analyzed_words.append({
                    "word": token.text,
                    "pos": token.pos_,  # Coarse-grained part-of-speech tag
                    "tag": token.tag_,  # Fine-grained part-of-speech tag
                    "ent_type": token.ent_type_, # Entity type
                    "is_stop": token.is_stop, # Whether the token is a stop word
                    "original_item_id": item.get("item_id"), # Link back to the original item
                    "original_score": item.get("score") # Original attribution score
                })
                pos_counts[token.pos_] += 1
                tag_counts[token.tag_] += 1
                label_counts[token.ent_type_] += 1

        output_file = pools_base_path / f"pos_analysis_{signal}_{model_key}.jsonl"
        save_jsonl(output_file, analyzed_words)

        pos_analysis_results[signal] = {
            "total_analyzed_words": len(analyzed_words),
            "pos_counts": dict(pos_counts),
            "tag_counts": dict(tag_counts),
            "label_counts": dict(label_counts),
            "output_file": str(output_file)
        }
        print(f"Saved POS analysis for {signal}_{model_key} to {output_file}")

    return pos_analysis_results

def mine_errors(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "mining_results.json"

    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Initialize spaCy model once for the entire mining process
    _nlp_model = None
    try:
        _nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        _nlp_model = spacy.load("en_core_web_sm")

    for task in config.tasks:
        task_results = {}

        for model_name in config.models:
            model_key = model_name.replace("/", "_").replace("-", "_")
            if config.use_generative_mode:
                model_key = model_key + "_gen"
            print(f"Mining errors for {task} with {model_key}")

            mining_results = mine_errors_for_task(config, run_dir, task, model_key)

            pool_results = build_candidate_pools(config, run_dir, task, model_key)

            overlap_results = compute_pool_overlap(config, run_dir, task, model_key)

            pos_analysis_results = analyze_pos_tags(_nlp_model, config, run_dir, task, model_key)

            task_results[model_key] = {
                "mining": mining_results,
                "pools": pool_results,
                "overlaps": overlap_results,
                "pos_analysis": pos_analysis_results
            }

        results[task] = task_results

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    # Hardcoded parameters for testing
    task = "sst2"
    model = "distilbert-base-uncased"
    signals = ["ig", "attn"]
    foundation_dir = Path("foundation")
    use_generative = False

    model_key = model.replace("/", "_").replace("-", "_")
    if use_generative:
        model_key = model_key + "_gen"

    # Create a minimal config for overlap computation
    config = ExperimentConfig(
        tasks=[task],
        models=[model],
        signals=signals,
        use_generative_mode=use_generative
    )

    print(f"Computing pool overlap for task={task}, model={model_key}, signals={signals}")

    overlap_results = compute_pool_overlap(config, foundation_dir, task, model_key)

    # Save results
    output_file = foundation_dir / "pools" / task / f"overlap_analysis_{model_key}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(overlap_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"OVERLAP ANALYSIS RESULTS")
    print(f"{'=' * 60}\n")

    if overlap_results.get("error_vs_correct"):
        print("Error vs Correct (Same Signal):")
        for sig, stats in overlap_results["error_vs_correct"].items():
            print(f"  {sig}: Jaccard={stats['jaccard']:.3f}, Intersection={stats['intersection_size']}")

    if overlap_results.get("cross_signal_error"):
        print("\nCross-Signal Error Pool:")
        for pair, stats in overlap_results["cross_signal_error"].items():
            print(f"  {pair}: Jaccard={stats['jaccard']:.3f}, Intersection={stats['intersection_size']}")

    if overlap_results.get("cross_signal_correct"):
        print("\nCross-Signal Correct Pool:")
        for pair, stats in overlap_results["cross_signal_correct"].items():
            print(f"  {pair}: Jaccard={stats['jaccard']:.3f}, Intersection={stats['intersection_size']}")

    print(f"\n✓ Results saved to: {output_file}\n")
