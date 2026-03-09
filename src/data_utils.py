import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
import json
import torch
from .config import ExperimentConfig
from .utils import save_jsonl, load_jsonl

def load_task_data(task_name: str, max_items: int = None) -> Tuple[Dataset, Dataset, Dataset]:
    if task_name.lower() == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_data = dataset["train"]
        dev_data = dataset["validation"]
        test_data = dataset["test"] # Note: all labels in test data are -1

    elif task_name.lower() == "qnli":
        dataset = load_dataset("glue", "qnli")
        train_data = dataset["train"]
        dev_data = dataset["validation"]
        test_data = dataset["test"]
    elif task_name.lower() == "news":
        dataset = load_dataset("sh0416/ag_news")
        train_data = dataset["train"]
        dev_data = dataset["test"]
        test_data = dataset["test"]
    elif task_name.lower() == "github":
        csv_file = './Dataset/github-labels-top3-803k-train.csv'
        df = pd.read_csv(csv_file)

        df = df[['issue_label', 'issue_title', 'issue_body']]  # Keep only text and label columns

        # Convert labels to numerical values
        label_mapping = {'bug': 0, 'enhancement': 1, 'question': 2}
        df.rename(columns={'issue_label': 'label'}, inplace=True)
        df['label'] = df['label'].map(label_mapping)
        df = df.dropna(subset=['label'])  # Remove rows where label is NaN after mapping

        if max_items:
            df = df.head(max_items)

        dataset = Dataset.from_pandas(df)
        # print(df['label'].value_counts())
        # Split the dataset into training, development, and test sets (80/10/10 split)
        train_idx, test_dev_idx = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            random_state=42,
            stratify=dataset["label"]  # Add stratification here
        )
        train_data = dataset.select(train_idx)
        test_dev_data = dataset.select(test_dev_idx)

        dev_idx, test_idx = train_test_split(
            range(len(test_dev_idx)),
            test_size=0.5,
            random_state=42,
            stratify=test_dev_data["label"]  # Add stratification here
        )
        dev_data = test_dev_data.select(dev_idx)
        test_data = test_dev_data.select(test_idx)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    if max_items:
        train_data = train_data.select(range(min(max_items, len(train_data))))
        dev_data = dev_data.select(range(min(max_items // 10, len(dev_data))))
        test_data = test_data.select(range(min(max_items // 10, len(test_data))))

    return train_data, dev_data, test_data

def load_adversarial_data(task_name: str, max_items: int = None) -> Dataset:
    try:
        task_to_adv_config = {
            "sst2": "adv_sst2",
            "qnli": "adv_qnli",
            "mnli": "adv_mnli",
            "qqp": "adv_qqp",
            "rte": "adv_rte"
        }

        adv_config_name = task_to_adv_config.get(task_name.lower())
        if not adv_config_name:
            print(f"Warning: No adversarial config found for task {task_name}")
            return None

        adv_dataset = load_dataset("AI-Secure/adv_glue", adv_config_name)

        if "validation" in adv_dataset:
            adv_dev_data = adv_dataset["validation"]
        elif "test" in adv_dataset:
            adv_dev_data = adv_dataset["test"]
        else:
            raise ValueError(f"No suitable split found in adversarial dataset for {task_name}")

        if max_items:
            adv_dev_data = adv_dev_data.select(range(min(max_items // 10, len(adv_dev_data))))

        return adv_dev_data

    except Exception as e:
        print(f"Warning: Could not load adversarial data for {task_name}: {e}")
        return None

def split_dataset(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    results = {}

    for task in config.tasks:
        train_data, dev_data, test_data = load_task_data(
            task,
            max_items=config.max_items if config.debug else None
        )

        train_indices = list(range(len(train_data)))
        np.random.seed(config.seed)

        fit_indices, mining_indices = train_test_split(
            train_indices,
            test_size=config.mining_split_ratio,
            random_state=config.seed,
            stratify=[train_data[i]["label"] for i in train_indices]
        )

        fit_data = train_data.select(fit_indices)
        mining_data = train_data.select(mining_indices)

        fit_items = convert_to_standard_format(fit_data, task, fit_indices)
        mining_items = convert_to_standard_format(mining_data, task, mining_indices)
        dev_items = convert_to_standard_format(dev_data, task)
        test_items = convert_to_standard_format(test_data, task)

        task_dir = run_dir / "data" / task
        task_dir.mkdir(parents=True, exist_ok=True)

        save_jsonl(task_dir / "fit.jsonl", fit_items)
        save_jsonl(task_dir / "mining.jsonl", mining_items)
        save_jsonl(task_dir / "dev.jsonl", dev_items)
        save_jsonl(task_dir / "test.jsonl", test_items)

        def get_label_dist(items):
            labels = [item["label"] for item in items]
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique.tolist(), counts.tolist()))
        
        results[task] = {
            "splits": {
                "fit": len(fit_items),
                "mining": len(mining_items),
                "dev": len(dev_items),
                "test": len(test_items)
            },
            "label_distributions": {
                "fit": get_label_dist(fit_items),
                "mining": get_label_dist(mining_items),
                "dev": get_label_dist(dev_items),
                "test": get_label_dist(test_items),
                "overall": get_label_dist(fit_items + mining_items + dev_items + test_items)
            }
        }

    with open(run_dir / "results" / "split_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

def convert_to_standard_format(dataset: Dataset, task: str, indices: List[int] = None) -> List[Dict[str, Any]]:
    items = []

    for i, item in enumerate(dataset):
        original_idx = indices[i] if indices is not None else item.get("idx", i)
        
        if task.lower() == "sst2":
            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": item["sentence"],
                "text_b": None,
                "label": item["label"]
            }
        elif task.lower() == "qnli":
            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": item["question"],
                "text_b": item["sentence"],
                "label": item["label"]
            }
        elif task.lower() == "news":
            # print("news item example",item)
            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": item["title"],
                "text_b": item["description"],
                "label": item["label"] - 1
            }
        elif task.lower() == "github":
            text_a = str(item["issue_title"]) if item["issue_title"] is not None else ""
            text_b = str(item["issue_body"]) if item["issue_body"] is not None else ""

            standard_item = {
                "id": f"{task}_{original_idx}",
                "text_a": text_a,
                "text_b": text_b,
                "label": item["label"]
            }
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        items.append(standard_item)
    
    return items

def load_split_data(run_dir: Path, task: str, split: str) -> List[Dict[str, Any]]:
    filepath = run_dir / "data" / task / f"{split}.jsonl"
    return load_jsonl(filepath)

def get_label_names(task: str) -> List[str]:
    if task.lower() == "sst2":
        return ["negative", "positive"]
    elif task.lower() == "qnli":
        return ["entailment", "not_entailment"]
    elif task.lower() == "news":
        return ["World", "Sports", "Business", "Sci/Tech"]
    elif task.lower() == "github":
        return ['bug', 'enhancement', 'question']
    else:
        raise ValueError(f"Unsupported task: {task}")

def get_task_prompt(task: str,  text_a, text_b) -> str:
    label_names = get_label_names(task)
    if task.lower() == "sst2":
        # SST-2: sentiment analysis
        prompt = f"Instruction: Classify the sentiment of the following sentence as {', '.join(label_names)}.\nInput: {text_a}\nSentiment:"

    elif task.lower() == "qnli":
        # QNLI: entailment  not_entailment
        prompt = f"Instruction: Determine if the following context contains the answer to the question. Answer with {' or '.join(label_names)}.\nQuestion: {text_a}\nContext: {text_b}\n"
    elif task.lower() == "news":
        # AG News: classify news articles into World, Sports, Business, or Sci/Tech
        prompt = f"Instruction: Classify the following news article into only one of four categories: {', '.join(label_names)}.\nTitle： {text_a}Description: {text_b}\nCategory:"
    elif task.lower() == "github":
        prompt = f"Instruction: Classify the following github issue into one of three categories: {', '.join(label_names)}.\nTitle： {text_a}.\nBody: {text_b}\nCategory:"
        prompt_1 = f"""<|im_start|>system
You are a helpful assistant that classifies GitHub issues.<|im_end|>
<|im_start|>user
Classify the following github issue into one of three categories: bug, enhancement, question.
Title: {text_a}
Body: {text_b}
Category:<|im_end|>
<|im_start|>assistant
"""
        prompt2 = f"""Instruction:
Classify the following GitHub issue into one of these categories: {' or '.join(label_names)}.
Title: {text_a}
Body: {text_b}
Category:
"""
    else:
        # general template
        input_str = f"{text_a} {text_b}" if text_b else text_a
        prompt = f"Instruction: Process the following task.\nInput: {input_str}\nOutput only one of following words: {', '.join(label_names)}\nAnswer:"
    return prompt

def prepare_model_inputs(items: List[Dict[str, Any]], tokenizer, max_length: int = 128,
                         is_generative: bool = False, task: str = "sst2"):
    # texts_a = [item["text_a"] for item in items]
    # texts_b = [item.get("text_b") for item in items]
    # if task.lower() == "github":
    #     max_length = 1024
    # else:
    #     max_length = 512
    
    # 1. Extract safe EOS variable to prevent NoneType errors
    safe_eos = tokenizer.eos_token or ""

    # 2. Modify the handling of texts_a
    texts_a = [
        str(item["text_a"]).replace(safe_eos, "")
        for item in items
        if item.get("text_a") is not None
    ]

    # 3. Modify the handling logic of texts_b
    raw_texts_b = [item.get("text_b") for item in items]

    if all(b is None for b in raw_texts_b):
        texts_b = []
    else:
        # Unified processing: convert to string -> replace EOS -> ensure not None
        texts_b = [
            str(b).replace(safe_eos, "") if b is not None else ""
            for b in raw_texts_b
        ]
    tokenizer_kwargs = {"truncation": True, "padding": True, "max_length": max_length, "return_tensors": "pt"}

    if not is_generative:
        # texts_a = [item["text_a"] for item in items]
        # texts_b = [item["text_b"] for item in items] if items and items[0]["text_b"] is not None else []
        if len(texts_b) == 0:
            encodings = tokenizer(texts_a, **tokenizer_kwargs)
        else:
            encodings = tokenizer(texts_a,texts_b, **tokenizer_kwargs)
        if 'input_ids' in encodings:
            encodings['input_ids'] = encodings['input_ids'].long()

        labels = [item["label"] for item in items]
        return encodings, labels
    else:
        # For generative LLM (Causal LM)
        label_names = get_label_names(task) # Get the list, such as ['negative', 'positive']

        all_input_ids = []
        all_labels = []
        full_texts = []
        for item in items:
            text_a = item["text_a"]
            text_b = item.get("text_b", "")
            label_str = label_names[item["label"]]

            prompt = get_task_prompt(task, text_a, text_b)
            full_text = f"{prompt} {label_str}{tokenizer.eos_token}"

            prompt_encoded = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
            full_encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)

            prompt_len = len(prompt_encoded["input_ids"])

            # 3. construct input_ids and labels
            # Enter -100 in the prompt field of the labels;
            # it will be automatically ignored when calculating the loss.
            input_ids = full_encoded["input_ids"]
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            all_input_ids.append(torch.tensor(input_ids))
            all_labels.append(torch.tensor(labels))

        # Right padding for training
        tokenizer.padding_side = "right"
        def pad_sequence(sequences, padding_value):
            return torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=padding_value
            )

        padded_input_ids = pad_sequence(all_input_ids, tokenizer.pad_token_id or 0)
        padded_labels = pad_sequence(all_labels, -100)

        # Build attention mask
        attention_mask = (padded_input_ids != (tokenizer.pad_token_id or 0)).long()

        encodings = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
        }
        # Cut off the sequence if it exceeds max_length
        if padded_input_ids.shape[1] > max_length:
            encodings["input_ids"] = encodings["input_ids"][:, :max_length]
            encodings["attention_mask"] = encodings["attention_mask"][:, :max_length]
            padded_labels = padded_labels[:, :max_length]

        return encodings, padded_labels

        # Use .clone() to ensure modifying labels doesn't affect original data
        encodings["labels"] = encodings["input_ids"].clone()

        # Find all padding locations
        # Assume tokenizer.pad_token_id is 15
        print("Example of first 2 prompts and labels")
        for i in range(2):
            print(f"Prompt: {full_texts[i]}\n  {encodings}")
        return encodings, encodings["input_ids"].clone()
