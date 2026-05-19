#!/usr/bin/env python3
"""
Analyze ALL server-saved survey JSON submissions, including MTurk workers who did not
paste a completion code and internal colleagues. Automatically detects the gold / attention
question in each batch and excludes submissions that fail it.

Usage:
  python3 scripts/analyze_all_results_attention.py \
    --results-dir results/server_saved \
    --batches-dir public/batches \
    --outdir results/attention_filtered

Optional, if you also downloaded MTurk results CSV:
  python3 scripts/analyze_all_results_attention.py \
    --results-dir results/server_saved \
    --batches-dir public/batches \
    --mturk-csv mturk_results.csv \
    --outdir results/attention_filtered
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

CODE_RE = re.compile(r"SGV-[A-Za-z0-9_.-]+-[A-Z0-9]{8}")
BACKUP_PREFIX = "SGVJSON-"


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def is_gold_item(item_id: str) -> bool:
    return str(item_id or "").startswith("gold_")


def expected_for_gold(item_id: str) -> Tuple[str, str]:
    """Return expected (label_preserved, fluent) for our generated gold items."""
    item_id = str(item_id or "").lower()
    if "label_no" in item_id:
        return "no", "yes"
    if "fluent_no" in item_id or "fluency_no" in item_id:
        return "yes", "no"
    # gold_*_yes_* means clearly label-preserved and fluent.
    return "yes", "yes"


def load_batches(batches_dir: pathlib.Path) -> Dict[str, dict]:
    batches: Dict[str, dict] = {}
    for p in sorted(batches_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        bid = str(obj.get("batch_id") or p.stem)
        batches[bid] = obj
    return batches


def load_json_results(results_dir: pathlib.Path) -> List[dict]:
    rows: List[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"_file": str(p), "_load_error": str(e)})
            continue
        obj["_file"] = str(p)
        rows.append(obj)
    return rows


def find_code_in_mturk_row(row: dict) -> str:
    keys = [k for k in row if k.lower().startswith("answer.")] + list(row.keys())
    seen = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        val = str(row.get(k, "") or "").strip()
        if not val:
            continue
        if val.startswith(BACKUP_PREFIX):
            return val
        m = CODE_RE.search(val)
        if m:
            return m.group(0)
    return ""


def decode_backup_code(code: str) -> dict | None:
    try:
        raw = base64.b64decode(code[len(BACKUP_PREFIX):].encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def load_mturk_code_map(mturk_csv: pathlib.Path | None) -> Dict[str, dict]:
    """Map completion code -> MTurk metadata. Empty if no MTurk CSV provided."""
    if not mturk_csv:
        return {}
    mapping: Dict[str, dict] = {}
    if not mturk_csv.exists():
        print(f"Warning: MTurk CSV not found: {mturk_csv}")
        return mapping
    with mturk_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = find_code_in_mturk_row(row)
            if not code:
                continue
            mapping[code] = {
                "MTurk_AssignmentId": row.get("AssignmentId", row.get("assignmentId", "")),
                "MTurk_WorkerId": row.get("WorkerId", row.get("workerId", "")),
                "MTurk_HITId": row.get("HITId", row.get("hitId", "")),
                "MTurk_AnswerCode": code,
            }
    return mapping


def response_by_item(payload: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in payload.get("responses", []) or []:
        iid = str(r.get("item_id", ""))
        if iid:
            out[iid] = r
    return out


def evaluate_attention(payload: dict, batch: dict | None) -> dict:
    """Evaluate gold item(s). Returns summary dict."""
    if not isinstance(payload, dict):
        return {"attention_status": "invalid_payload", "attention_pass": False, "attention_total": 0, "attention_correct": 0}
    if not batch:
        return {"attention_status": "missing_batch", "attention_pass": False, "attention_total": 0, "attention_correct": 0}

    gold_items = [it for it in batch.get("items", []) if is_gold_item(str(it.get("item_id", "")))]
    if not gold_items:
        return {"attention_status": "no_attention_item_in_batch", "attention_pass": False, "attention_total": 0, "attention_correct": 0}

    responses = response_by_item(payload)
    details = []
    correct = 0
    total = 0
    for it in gold_items:
        iid = str(it.get("item_id", ""))
        exp_label, exp_fluent = expected_for_gold(iid)
        r = responses.get(iid)
        if not r:
            details.append(f"{iid}: missing response, expected label={exp_label}, fluent={exp_fluent}")
            total += 1
            continue
        got_label = norm(r.get("label_preserved"))
        got_fluent = norm(r.get("fluent"))
        ok = (got_label == exp_label and got_fluent == exp_fluent)
        if ok:
            correct += 1
        total += 1
        details.append(f"{iid}: got label={got_label or '[blank]'}, fluent={got_fluent or '[blank]'}; expected label={exp_label}, fluent={exp_fluent}; {'OK' if ok else 'FAIL'}")

    passed = (total > 0 and correct == total)
    return {
        "attention_status": "pass" if passed else "fail",
        "attention_pass": passed,
        "attention_total": total,
        "attention_correct": correct,
        "attention_details": " | ".join(details),
    }


def write_csv(path: pathlib.Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze all server-saved results and exclude failed attention checks.")
    ap.add_argument("--results-dir", default="results/server_saved", help="Directory containing server-saved JSON submissions")
    ap.add_argument("--batches-dir", default="public/batches", help="Directory containing batch JSON files")
    ap.add_argument("--mturk-csv", default="", help="Optional downloaded MTurk results CSV; used only to attach WorkerId/AssignmentId when a code exists")
    ap.add_argument("--outdir", default="results/attention_filtered", help="Output directory")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    results_dir = pathlib.Path(args.results_dir); results_dir = results_dir if results_dir.is_absolute() else root / results_dir
    batches_dir = pathlib.Path(args.batches_dir); batches_dir = batches_dir if batches_dir.is_absolute() else root / batches_dir
    outdir = pathlib.Path(args.outdir); outdir = outdir if outdir.is_absolute() else root / outdir
    mturk_csv = pathlib.Path(args.mturk_csv) if args.mturk_csv else None
    if mturk_csv and not mturk_csv.is_absolute():
        mturk_csv = root / mturk_csv

    batches = load_batches(batches_dir)
    submissions = load_json_results(results_dir)
    mturk_by_code = load_mturk_code_map(mturk_csv)

    assignment_rows: List[dict] = []
    item_rows_all: List[dict] = []
    item_rows_valid: List[dict] = []
    gold_rows: List[dict] = []

    seen_server_codes = set()
    for idx, payload in enumerate(submissions, start=1):
        if "_load_error" in payload:
            assignment_rows.append({
                "submission_index": idx,
                "source_file": payload.get("_file", ""),
                "exclude_reason": "json_load_error",
                "attention_status": "invalid_json",
                "attention_pass": False,
                "json_load_error": payload.get("_load_error", ""),
            })
            continue

        batch_id = str(payload.get("batch_id", ""))
        batch = batches.get(batch_id)
        att = evaluate_attention(payload, batch)
        server_code = str(payload.get("server_code") or payload.get("client_code") or "")
        seen_server_codes.add(server_code)
        mturk_meta = mturk_by_code.get(server_code, {}) if server_code else {}

        exclude_reason = "" if att["attention_pass"] else "failed_attention_check"
        if att.get("attention_status") == "missing_batch":
            exclude_reason = "missing_batch"
        elif att.get("attention_status") == "invalid_payload":
            exclude_reason = "invalid_payload"

        assn = {
            "submission_index": idx,
            "source_file": payload.get("_file", ""),
            "server_code": server_code,
            "batch_id": batch_id,
            "task": payload.get("task", ""),
            "worker_id_entered": payload.get("worker_id_entered", ""),
            "submitted_at": payload.get("submitted_at", ""),
            "server_saved_at": payload.get("server_saved_at", ""),
            "attention_status": att.get("attention_status", ""),
            "attention_pass": att.get("attention_pass", False),
            "attention_correct": att.get("attention_correct", 0),
            "attention_total": att.get("attention_total", 0),
            "attention_details": att.get("attention_details", ""),
            "exclude_reason": exclude_reason,
            "json_load_error": "",
            **mturk_meta,
        }
        assignment_rows.append(assn)

        for r in payload.get("responses", []) or []:
            iid = str(r.get("item_id", ""))
            is_gold = is_gold_item(iid)
            reasons = r.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = []
            row = {
                "submission_index": idx,
                "server_code": server_code,
                "batch_id": batch_id,
                "task": r.get("task", payload.get("task", "")),
                "item_id": iid,
                "is_gold": is_gold,
                "true_label": r.get("true_label", ""),
                "label_preserved": r.get("label_preserved", ""),
                "fluent": r.get("fluent", ""),
                "reasons": ";".join(reasons),
                "note": r.get("note", ""),
                "time_ms": r.get("time_ms", ""),
                "attention_pass_for_submission": att.get("attention_pass", False),
                "exclude_reason": exclude_reason,
                "worker_id_entered": payload.get("worker_id_entered", ""),
                **mturk_meta,
            }
            item_rows_all.append(row)
            if is_gold:
                exp_label, exp_fluent = expected_for_gold(iid)
                row_g = dict(row)
                row_g.update({"expected_label_preserved": exp_label, "expected_fluent": exp_fluent})
                gold_rows.append(row_g)
            elif att.get("attention_pass", False):
                item_rows_valid.append(row)

    # Also report MTurk codes that had no matching server JSON, if MTurk CSV was provided.
    for code, meta in mturk_by_code.items():
        if code and code not in seen_server_codes:
            # If it is a backup code, decode and include it as a submission-like record.
            decoded = decode_backup_code(code) if code.startswith(BACKUP_PREFIX) else None
            if decoded:
                batch_id = str(decoded.get("batch_id", ""))
                batch = batches.get(batch_id)
                att = evaluate_attention(decoded, batch)
                exclude_reason = "" if att["attention_pass"] else "failed_attention_check"
                idx = len(assignment_rows) + 1
                assignment_rows.append({
                    "submission_index": idx,
                    "source_file": "decoded_from_mturk_backup_code",
                    "server_code": code[:80] + "..." if len(code) > 80 else code,
                    "batch_id": batch_id,
                    "task": decoded.get("task", ""),
                    "worker_id_entered": decoded.get("worker_id_entered", ""),
                    "submitted_at": decoded.get("submitted_at", ""),
                    "server_saved_at": "",
                    "attention_status": att.get("attention_status", ""),
                    "attention_pass": att.get("attention_pass", False),
                    "attention_correct": att.get("attention_correct", 0),
                    "attention_total": att.get("attention_total", 0),
                    "attention_details": att.get("attention_details", ""),
                    "exclude_reason": exclude_reason,
                    "json_load_error": "",
                    **meta,
                })
                for r in decoded.get("responses", []) or []:
                    iid = str(r.get("item_id", "")); is_gold = is_gold_item(iid)
                    reasons = r.get("reasons", []) if isinstance(r.get("reasons", []), list) else []
                    row = {
                        "submission_index": idx,
                        "server_code": code[:80] + "..." if len(code) > 80 else code,
                        "batch_id": batch_id,
                        "task": r.get("task", decoded.get("task", "")),
                        "item_id": iid,
                        "is_gold": is_gold,
                        "true_label": r.get("true_label", ""),
                        "label_preserved": r.get("label_preserved", ""),
                        "fluent": r.get("fluent", ""),
                        "reasons": ";".join(reasons),
                        "note": r.get("note", ""),
                        "time_ms": r.get("time_ms", ""),
                        "attention_pass_for_submission": att.get("attention_pass", False),
                        "exclude_reason": exclude_reason,
                        "worker_id_entered": decoded.get("worker_id_entered", ""),
                        **meta,
                    }
                    item_rows_all.append(row)
                    if is_gold:
                        exp_label, exp_fluent = expected_for_gold(iid)
                        row_g = dict(row); row_g.update({"expected_label_preserved": exp_label, "expected_fluent": exp_fluent})
                        gold_rows.append(row_g)
                    elif att.get("attention_pass", False):
                        item_rows_valid.append(row)
            else:
                assignment_rows.append({
                    "submission_index": len(assignment_rows) + 1,
                    "source_file": "mturk_csv_only_no_server_json",
                    "server_code": code,
                    "batch_id": "",
                    "task": "",
                    "worker_id_entered": "",
                    "submitted_at": "",
                    "server_saved_at": "",
                    "attention_status": "no_server_json_found",
                    "attention_pass": False,
                    "attention_correct": 0,
                    "attention_total": 0,
                    "attention_details": "Completion code appears in MTurk CSV but no matching JSON was found in results/server_saved.",
                    "exclude_reason": "no_server_json_found",
                    "json_load_error": "",
                    **meta,
                })

    # Simple majority summary from valid, non-gold rows.
    by_item: Dict[str, List[dict]] = defaultdict(list)
    for row in item_rows_valid:
        by_item[str(row.get("item_id", ""))].append(row)
    majority_rows: List[dict] = []
    for item_id, rows in sorted(by_item.items()):
        def maj(field: str) -> str:
            vals = [norm(r.get(field)) for r in rows if norm(r.get(field))]
            yes = vals.count("yes"); no = vals.count("no"); ct = vals.count("cannot_tell") + vals.count("cannot tell")
            if yes > no and yes > ct: return "yes"
            if no > yes and no > ct: return "no"
            if ct > yes and ct > no: return "cannot_tell"
            return "tie_or_no_majority"
        m_label = maj("label_preserved")
        m_fluent = maj("fluent")
        majority_rows.append({
            "item_id": item_id,
            "task": rows[0].get("task", "") if rows else "",
            "batch_id": rows[0].get("batch_id", "") if rows else "",
            "true_label": rows[0].get("true_label", "") if rows else "",
            "n_valid_annotations": len(rows),
            "label_preserved_majority": m_label,
            "fluent_majority": m_fluent,
            "human_valid_majority": (m_label == "yes" and m_fluent == "yes"),
            "label_yes_count": sum(1 for r in rows if norm(r.get("label_preserved")) == "yes"),
            "label_no_count": sum(1 for r in rows if norm(r.get("label_preserved")) == "no"),
            "label_cannot_tell_count": sum(1 for r in rows if norm(r.get("label_preserved")) in {"cannot_tell", "cannot tell"}),
            "fluent_yes_count": sum(1 for r in rows if norm(r.get("fluent")) == "yes"),
            "fluent_no_count": sum(1 for r in rows if norm(r.get("fluent")) == "no"),
            "fluent_cannot_tell_count": sum(1 for r in rows if norm(r.get("fluent")) in {"cannot_tell", "cannot tell"}),
        })

    assignment_fields = [
        "submission_index", "source_file", "server_code", "batch_id", "task", "worker_id_entered",
        "MTurk_AssignmentId", "MTurk_WorkerId", "MTurk_HITId", "MTurk_AnswerCode",
        "submitted_at", "server_saved_at", "attention_status", "attention_pass", "attention_correct",
        "attention_total", "attention_details", "exclude_reason", "json_load_error",
    ]
    item_fields = [
        "submission_index", "server_code", "batch_id", "task", "item_id", "is_gold", "true_label",
        "label_preserved", "fluent", "reasons", "note", "time_ms", "attention_pass_for_submission",
        "exclude_reason", "worker_id_entered", "MTurk_AssignmentId", "MTurk_WorkerId", "MTurk_HITId",
    ]
    majority_fields = [
        "item_id", "task", "batch_id", "true_label", "n_valid_annotations", "label_preserved_majority",
        "fluent_majority", "human_valid_majority", "label_yes_count", "label_no_count",
        "label_cannot_tell_count", "fluent_yes_count", "fluent_no_count", "fluent_cannot_tell_count",
    ]
    gold_fields = item_fields + ["expected_label_preserved", "expected_fluent"]

    write_csv(outdir / "assignment_attention_summary.csv", assignment_rows, assignment_fields)
    write_csv(outdir / "item_annotations_all.csv", item_rows_all, item_fields)
    write_csv(outdir / "item_annotations_valid_only.csv", item_rows_valid, item_fields)
    write_csv(outdir / "gold_attention_details.csv", gold_rows, gold_fields)
    write_csv(outdir / "item_majority_valid_only.csv", majority_rows, majority_fields)

    total = len(assignment_rows)
    passed = sum(1 for r in assignment_rows if str(r.get("attention_pass")) == "True" or r.get("attention_pass") is True)
    failed = total - passed
    print(f"Read server submissions: {len(submissions)}")
    print(f"Total assignment-like records: {total}")
    print(f"Attention pass: {passed}")
    print(f"Excluded/fail/missing: {failed}")
    print(f"Valid non-gold item rows: {len(item_rows_valid)}")
    print(f"Wrote outputs to: {outdir}")
    print("Main files:")
    print(f"  {outdir / 'assignment_attention_summary.csv'}")
    print(f"  {outdir / 'item_annotations_valid_only.csv'}")
    print(f"  {outdir / 'item_majority_valid_only.csv'}")


if __name__ == "__main__":
    main()
