#!/usr/bin/env python3
"""
Fix/enrich human annotation rows with original generation metadata
(explainer, strategy, operation, model, word, flipped).

This version maps using columns available in human_metric_joined/enriched output:
task, true_label, sample_type, ppl_diff, log_ppl_diff, nli_forward, nli_backward.
It does not require a row index.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


LABEL_MAPS = {
    "github": {0: "bug", 1: "enhancement", 2: "question"},
    "news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "sst2": {0: "negative", 1: "positive"},
}

def norm_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def norm_task(x):
    s = norm_text(x)
    if s in {"ag news", "ag_news"}:
        return "news"
    return s

def norm_label(task, label):
    if pd.isna(label):
        return ""
    t = norm_task(task)
    raw = str(label).strip()
    raw_low = raw.lower()
    known = {
        "bug": "bug",
        "enhancement": "enhancement",
        "question": "question",
        "positive": "positive",
        "negative": "negative",
        "world": "World",
        "sports": "Sports",
        "business": "Business",
        "sci/tech": "Sci/Tech",
        "science/technology": "Sci/Tech",
        "science and technology": "Sci/Tech",
    }
    if raw_low in known:
        return known[raw_low]
    try:
        i = int(float(raw))
        if t in LABEL_MAPS and i in LABEL_MAPS[t]:
            return LABEL_MAPS[t][i]
    except Exception:
        pass
    return raw

def norm_bool(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return pd.NA

def add_key_cols(df):
    out = df.copy()
    if "task" not in out.columns:
        out["task"] = ""
    if "true_label" not in out.columns:
        out["true_label"] = ""
    if "sample_type" not in out.columns:
        out["sample_type"] = ""

    out["__task_key"] = out["task"].map(norm_task)
    out["__label_key"] = [norm_text(norm_label(t, y)) for t, y in zip(out["task"], out["true_label"])]
    out["__sample_key"] = out["sample_type"].map(norm_text)

    for c in ["ppl_diff", "log_ppl_diff", "nli_forward", "nli_backward"]:
        if c in out.columns:
            out[f"__{c}_key"] = pd.to_numeric(out[c], errors="coerce").round(8)
        else:
            out[f"__{c}_key"] = np.nan
    return out

def rate(x):
    vals = x.dropna().map(norm_bool).dropna()
    if len(vals) == 0:
        return np.nan
    return vals.astype(bool).mean()

def summarize_by(df, group_col):
    d = df.copy()
    rows = []
    for value, g in d.groupby(group_col, dropna=False):
        rows.append({
            group_col: value,
            "n_items": len(g),
            "human_valid_rate": rate(g["human_valid_majority"]) if "human_valid_majority" in g.columns else np.nan,
            "label_preserved_rate": rate(g["label_preserved_majority"]) if "label_preserved_majority" in g.columns else np.nan,
            "fluent_rate": rate(g["fluent_majority"]) if "fluent_majority" in g.columns else np.nan,
            "metric_accept_rate": rate(g["metric_accepts"]) if "metric_accepts" in g.columns else (
                (g["sample_type"].astype(str).str.lower() == "accepted").mean() if "sample_type" in g.columns else np.nan
            ),
        })
    out = pd.DataFrame(rows)
    if "human_valid_rate" in out.columns:
        out = out.sort_values(["human_valid_rate", "n_items"], ascending=[False, False])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-data", required=True, help="CSV with explainer/strategy/operation")
    ap.add_argument("--human-joined", required=True, help="human_metric_joined.csv or enriched_human_with_original_metadata.csv")
    ap.add_argument("--outdir", default="results/original_factor_analysis_fixed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    original_pd = pd.read_csv(args.original_data)
    mask = (
        original_pd["operation"].astype(str).str.lower().eq("ablate")
        & original_pd["strategy"].astype(str).str.lower().eq("random")
    )
    original_pd.loc[mask, "strategy"] = "insitu"

    original = add_key_cols(original_pd)
    human = add_key_cols(pd.read_csv(args.human_joined))

    key_cols = [
        "__task_key", "__label_key", "__sample_key",
        "__ppl_diff_key", "__log_ppl_diff_key", "__nli_forward_key", "__nli_backward_key",
    ]

    # If ppl_diff is missing in either file, fall back to log/nli only.
    if human["__ppl_diff_key"].isna().all() or original["__ppl_diff_key"].isna().all():
        key_cols = [
            "__task_key", "__label_key", "__sample_key",
            "__log_ppl_diff_key", "__nli_forward_key", "__nli_backward_key",
        ]

    meta_cols = [c for c in [
        "#", "model", "explainer", "strategy", "operation", "item_id",
        "word", "flipped", "is_accepted", "not_accepted"
    ] if c in original.columns]

    # One-to-one merge by occurrence for duplicate keys.
    h = human.copy()
    o = original.copy()
    h["__occ_key"] = h.groupby(key_cols, dropna=False).cumcount()
    o["__occ_key"] = o.groupby(key_cols, dropna=False).cumcount()

    merged = h.merge(
        o[key_cols + ["__occ_key"] + meta_cols],
        on=key_cols + ["__occ_key"],
        how="left",
        suffixes=("", "_orig"),
        indicator="__original_meta_merge",
    )

    # Human file may already have empty columns named explainer/strategy/operation.
    # Prefer original metadata suffix when present.
    for c in ["explainer", "strategy", "operation", "model", "word", "flipped", "is_accepted", "not_accepted", "#"]:
        alt = c + "_orig"
        if alt in merged.columns:
            if c in merged.columns:
                merged[c] = merged[alt].combine_first(merged[c])
                merged = merged.drop(columns=[alt])
            else:
                merged = merged.rename(columns={alt: c})

    if "item_id_orig" in merged.columns:
        merged = merged.rename(columns={"item_id_orig": "original_item_id"})

    matched = int((merged["__original_meta_merge"] == "both").sum())
    report = [
        f"Original rows: {len(original)}",
        f"Human rows: {len(human)}",
        f"Matched human rows to original metadata: {matched}/{len(human)}",
        f"Unmatched: {len(human) - matched}",
        "",
        "Non-null metadata counts after merge:",
    ]
    for c in ["explainer", "strategy", "operation", "model", "word", "flipped"]:
        if c in merged.columns:
            report.append(f"  {c}: {merged[c].notna().sum()}/{len(merged)}")
    report.append("")
    report.append("Key columns used:")
    report.extend([f"  {c}" for c in key_cols])

    print("\n".join(report))
    (outdir / "mapping_report.txt").write_text("\n".join(report), encoding="utf-8")

    out_csv = outdir / "enriched_human_with_original_metadata_fixed.csv"
    merged.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    for c in ["explainer", "strategy", "operation"]:
        if c in merged.columns:
            s = summarize_by(merged, c)
            out = outdir / f"summary_by_{c}.csv"
            s.to_csv(out, index=False)
            print("Wrote:", out)

    if all(c in merged.columns for c in ["explainer", "strategy", "operation"]):
        combo = merged.groupby(["explainer", "strategy", "operation"], dropna=False).apply(
            lambda g: pd.Series({
                "n_items": len(g),
                "human_valid_rate": rate(g["human_valid_majority"]) if "human_valid_majority" in g.columns else np.nan,
                "label_preserved_rate": rate(g["label_preserved_majority"]) if "label_preserved_majority" in g.columns else np.nan,
                "fluent_rate": rate(g["fluent_majority"]) if "fluent_majority" in g.columns else np.nan,
                "metric_accept_rate": rate(g["metric_accepts"]) if "metric_accepts" in g.columns else np.nan,
            })
        ).reset_index().sort_values(["human_valid_rate", "n_items"], ascending=[False, False])
        out = outdir / "summary_by_explainer_strategy_operation.csv"
        combo.to_csv(out, index=False)
        print("Wrote:", out)

if __name__ == "__main__":
    main()
