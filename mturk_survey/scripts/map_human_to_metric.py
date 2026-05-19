import argparse
import json
from pathlib import Path

import pandas as pd


def norm_bool(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return pd.NA


def build_key_from_batches(batches_dir):
    rows = []
    batches_dir = Path(batches_dir)

    for path in sorted(batches_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            batch = json.load(f)

        batch_id = batch.get("batch_id", path.stem)
        task = batch.get("task", "")

        for item in batch.get("items", []):
            item_id = item.get("item_id", "")
            if not item_id or item_id.startswith("gold_"):
                continue

            row = {
                "item_id": item_id,
                "batch_id": batch_id,
                "task_from_batch": item.get("task", task),
                "true_label_from_batch": item.get("true_label", item.get("label", "")),
                "original_title": item.get("original_title", ""),
                "original_body": item.get("original_body", ""),
                "mutated_title": item.get("mutated_title", ""),
                "mutated_body": item.get("mutated_body", ""),
            }

            # Keep any original row identifiers if they exist
            for k in [
                "source_row_id",
                "row_id",
                "original_index",
                "csv_index",
                "id",
                "sample_type",
                "accepted",
                "status",
                "gate_decision",
                "nli_score",
                "ppl_score",
                "semantic_score",
            ]:
                if k in item:
                    row[k] = item[k]

            rows.append(row)

    return pd.DataFrame(rows)

def print_binary_eval(df, pred_col, human_col, title):
    valid_mask = (
        df[pred_col].notna()
        & df[human_col].notna()
    )

    eval_df = df.loc[valid_mask].copy()

    print(f"\n=== {title} ===")
    print("Evaluated rows:", len(eval_df))

    if len(eval_df) == 0:
        print("No valid rows to evaluate.")
        return

    eval_df[pred_col] = eval_df[pred_col].astype(bool)
    eval_df[human_col] = eval_df[human_col].astype(bool)

    agree = (eval_df[pred_col] == eval_df[human_col]).mean()
    print("Agreement rate:", agree)

    print("\nConfusion table:")
    print(pd.crosstab(
        eval_df[pred_col],
        eval_df[human_col],
        rownames=[pred_col],
        colnames=[human_col],
        dropna=False,
    ))

    pred_true = eval_df[eval_df[pred_col] == True]
    pred_false = eval_df[eval_df[pred_col] == False]

    if len(pred_true) > 0:
        print(f"Human-positive rate among {pred_col}=True:",
              pred_true[human_col].mean())
    if len(pred_false) > 0:
        print(f"Human-positive rate among {pred_col}=False:",
              pred_false[human_col].mean())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--human-majority", required=True)
    parser.add_argument("--batches-dir", default="public/batches")
    parser.add_argument("--private-key", default=None)
    parser.add_argument("--out", default="results/human_metric_joined.csv")
    args = parser.parse_args()

    original = pd.read_csv(args.original_csv)
    human = pd.read_csv(args.human_majority)

    # Normalize human majority columns
    for c in ["label_preserved_majority", "fluent_majority", "human_valid_majority"]:
        if c in human.columns:
            human[c] = human[c].map(norm_bool)

    # Build or load item key
    if args.private_key and Path(args.private_key).exists():
        key = pd.read_csv(args.private_key)
    else:
        key = build_key_from_batches(args.batches_dir)

    print("Original rows:", len(original))
    print("Key rows:", len(key))
    print("Human majority rows:", len(human))

    if "item_id" not in key.columns:
        raise ValueError("Key file does not contain item_id")
    if "item_id" not in human.columns:
        raise ValueError("Human majority file does not contain item_id")

    # First join: human results + item key
    joined = human.merge(key, on="item_id", how="left", suffixes=("", "_key"))

    # Try to map back to original CSV
    # Best case: key contains original row index
    index_cols = [
        "source_row_id",
        "row_id",
        "original_index",
        "csv_index",
    ]

    used_mapping = None
    for col in index_cols:
        if col in joined.columns:
            # Convert to numeric index if possible
            tmp = joined.copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            if tmp[col].notna().sum() > 0:
                original_with_index = original.reset_index().rename(columns={"index": "__orig_index"})
                merged = tmp.merge(
                    original_with_index,
                    left_on=col,
                    right_on="__orig_index",
                    how="left",
                    suffixes=("", "_orig"),
                )
                if merged["__orig_index"].notna().sum() > 0:
                    joined = merged
                    used_mapping = f"original row index via {col}"
                    break

    # Fallback: text-based merge
    if used_mapping is None:
        print("No row-index mapping found. Trying text-based merge...")

        def make_text_key(df, title_col, body_col):
            return (
                df[title_col].fillna("").astype(str).str.strip()
                + "\n"
                + df[body_col].fillna("").astype(str).str.strip()
            ).str.replace(r"\s+", " ", regex=True).str.lower()

        # Guess original CSV columns
        possible_original_title = ["original_title", "orig_title", "title", "input_title"]
        possible_original_body = ["original_body", "orig_body", "body", "input_body", "text", "original_text"]
        possible_mutated_title = ["mutated_title", "mutation_title", "mut_title"]
        possible_mutated_body = ["mutated_body", "mutation_body", "mut_body", "mutated_text"]

        def pick(cols):
            for c in cols:
                if c in original.columns:
                    return c
            return None

        ot = pick(possible_original_title)
        ob = pick(possible_original_body)
        mt = pick(possible_mutated_title)
        mb = pick(possible_mutated_body)

        if ot and ob and mt and mb:
            orig2 = original.copy()
            orig2["__orig_text_key"] = make_text_key(orig2, ot, ob)
            orig2["__mut_text_key"] = make_text_key(orig2, mt, mb)

            joined["__orig_text_key"] = make_text_key(joined, "original_title", "original_body")
            joined["__mut_text_key"] = make_text_key(joined, "mutated_title", "mutated_body")

            joined = joined.merge(
                orig2,
                on=["__orig_text_key", "__mut_text_key"],
                how="left",
                suffixes=("", "_orig"),
            )
            used_mapping = "text-based original+mutated match"
        else:
            print("Could not guess original CSV text columns. Output will contain human + batch key only.")
            used_mapping = "human + batch key only"

    print("Mapping used:", used_mapping)

    # Create gate prediction columns if available
    # Adjust these names to your actual CSV columns.
    possible_gate_cols = [
        "sample_type",
        "gate_decision",
        "status",
        "accepted",
        "passed",
        "is_accepted",
    ]

    gate_col = None
    for c in possible_gate_cols:
        if c in joined.columns:
            gate_col = c
            break

    if gate_col:
        s = joined[gate_col].astype(str).str.lower()

        joined["metric_accepts"] = (
                s.str.contains("accept")
                | s.str.contains("passed")
                | s.eq("true")
                | s.eq("1")
        )

        # -----------------------------
        # 1. Metric valid vs human valid
        # human_valid_majority = label preserved AND fluent
        # -----------------------------
        if "human_valid_majority" in joined.columns:
            joined["metric_vs_human_valid_agree"] = (
                    joined["metric_accepts"] == joined["human_valid_majority"]
            )

            print("Gate column used:", gate_col)

            print("\n=== Metric valid vs human majority valid ===")
            print("Agreement rate:",
                  joined["metric_vs_human_valid_agree"].mean())

            print("\nConfusion table: metric_accepts vs human_valid_majority")
            print(pd.crosstab(
                joined["metric_accepts"],
                joined["human_valid_majority"],
                rownames=["metric_accepts"],
                colnames=["human_valid_majority"],
                dropna=False,
            ))

            # Useful rates
            accepted = joined[joined["metric_accepts"] == True]
            rejected = joined[joined["metric_accepts"] == False]

            if len(accepted) > 0:
                print("Accepted human-valid rate:",
                      accepted["human_valid_majority"].mean())
            if len(rejected) > 0:
                print("Rejected human-valid rate:",
                      rejected["human_valid_majority"].mean())

        else:
            print("Warning: human_valid_majority column not found.")

        # -----------------------------
        # 2. Metric valid vs human label preservation only
        # label_preserved_majority ignores fluency
        # -----------------------------
        if "label_preserved_majority" in joined.columns:
            # Convert to nullable boolean safely
            joined["label_preserved_majority_bool"] = joined["label_preserved_majority"].map(norm_bool)

            valid_lp_mask = (
                    joined["metric_accepts"].notna()
                    & joined["label_preserved_majority_bool"].notna()
            )

            joined["metric_vs_label_preserved_agree"] = pd.NA
            joined.loc[valid_lp_mask, "metric_vs_label_preserved_agree"] = (
                    joined.loc[valid_lp_mask, "metric_accepts"].astype(bool)
                    == joined.loc[valid_lp_mask, "label_preserved_majority_bool"].astype(bool)
            )

            lp_eval = joined.loc[valid_lp_mask].copy()

            print("\n=== Metric valid vs human label preservation only ===")
            print("Evaluated rows:", len(lp_eval))
            print("Agreement rate:", lp_eval["metric_vs_label_preserved_agree"].mean())

            print("\nConfusion table: metric_accepts vs label_preserved_majority")
            print(pd.crosstab(
                lp_eval["metric_accepts"].astype(bool),
                lp_eval["label_preserved_majority_bool"].astype(bool),
                rownames=["metric_accepts"],
                colnames=["label_preserved_majority"],
                dropna=False,
            ))

            accepted_lp = lp_eval[lp_eval["metric_accepts"] == True]
            rejected_lp = lp_eval[lp_eval["metric_accepts"] == False]

            if len(accepted_lp) > 0:
                print("Accepted label-preserved rate:",
                      accepted_lp["label_preserved_majority_bool"].mean())
            if len(rejected_lp) > 0:
                print("Rejected label-preserved rate:",
                      rejected_lp["label_preserved_majority_bool"].mean())

        else:
            print("Warning: label_preserved_majority column not found.")

        # ============================================================
        # Additional component-level threshold evaluations
        # ============================================================

        # Normalize human majority columns safely
        if "fluent_majority" in joined.columns:
            joined["fluent_majority_bool"] = joined["fluent_majority"].map(norm_bool)

        if "label_preserved_majority" in joined.columns:
            joined["label_preserved_majority_bool"] = joined["label_preserved_majority"].map(norm_bool)

        # -----------------------------
        # 3. PPL threshold vs human fluency only
        # metric_fluent = log_ppl_diff < 0.3
        # -----------------------------
        if "log_ppl_diff" in joined.columns and "fluent_majority_bool" in joined.columns:
            joined["log_ppl_diff_num"] = pd.to_numeric(joined["log_ppl_diff"], errors="coerce")
            joined["metric_fluent_ppl_lt_0_3"] = pd.NA

            valid_ppl = joined["log_ppl_diff_num"].notna()
            joined.loc[valid_ppl, "metric_fluent_ppl_lt_0_3"] = (
                    joined.loc[valid_ppl, "log_ppl_diff_num"] < 0.3
            )

            print_binary_eval(
                joined,
                pred_col="metric_fluent_ppl_lt_0_3",
                human_col="fluent_majority_bool",
                title="PPL fluency gate: log_ppl_diff < 0.3 vs human fluency majority",
            )
        else:
            print("\nWarning: Cannot evaluate PPL fluency gate. Need log_ppl_diff and fluent_majority.")

        # -----------------------------
        # 4. NLI thresholds vs human label preservation only
        # metric_label_preserved = nli_forward > 0.08 AND nli_backward > 0.08
        # -----------------------------
        if (
                "nli_forward" in joined.columns
                and "nli_backward" in joined.columns
                and "label_preserved_majority_bool" in joined.columns
        ):
            joined["nli_forward_num"] = pd.to_numeric(joined["nli_forward"], errors="coerce")
            joined["nli_backward_num"] = pd.to_numeric(joined["nli_backward"], errors="coerce")

            joined["metric_label_preserved_nli_gt_0_08"] = pd.NA

            valid_nli = (
                    joined["nli_forward_num"].notna()
                    & joined["nli_backward_num"].notna()
            )

            joined.loc[valid_nli, "metric_label_preserved_nli_gt_0_08"] = (
                    (joined.loc[valid_nli, "nli_forward_num"] > 0.08)
                    & (joined.loc[valid_nli, "nli_backward_num"] > 0.08)
            )

            print_binary_eval(
                joined,
                pred_col="metric_label_preserved_nli_gt_0_08",
                human_col="label_preserved_majority_bool",
                title="NLI label-preservation gate: nli_forward > 0.08 and nli_backward > 0.08 vs human label preservation majority",
            )
        else:
            print(
                "\nWarning: Cannot evaluate NLI label-preservation gate. Need nli_forward, nli_backward, and label_preserved_majority.")
    else:
        print("No gate/sample_type column found. Add mapping manually after inspecting columns.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out, index=False)
    print("Wrote:", out)


if __name__ == "__main__":
    main()