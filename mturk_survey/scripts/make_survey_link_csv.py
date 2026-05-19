#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, pathlib
from urllib.parse import quote

def main() -> None:
    ap = argparse.ArgumentParser(description='Create MTurk Survey Link CSV with one URL per batch.')
    ap.add_argument('--base-url', required=True, help='Public base URL, e.g. https://xxxx.trycloudflare.com')
    ap.add_argument('--batches-dir', default='public/batches')
    ap.add_argument('--out', default='data/mturk_survey_links_all.csv')
    ap.add_argument('--pilot', action='store_true', help='Only include first SST, AG, and GitHub batch')
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    batches_dir = (root / args.batches_dir).resolve()
    out = (root / args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip('/')
    rows = []
    for p in sorted(batches_dir.glob('*.json')):
        obj = json.loads(p.read_text(encoding='utf-8'))
        bid = obj['batch_id']
        rows.append({'batch_id': bid, 'task': obj.get('task',''), 'task_display': obj.get('task_display',''), 'real_item_count': obj.get('real_item_count',''), 'total_item_count': obj.get('total_item_count', len(obj.get('items',[]))), 'survey_url': f'{base}/survey.html?batch={quote(bid)}'})
    if args.pilot:
        picked=[]; seen=set()
        for task in ['sst2','ag_news','github']:
            for r in rows:
                if r['task']==task and task not in seen:
                    picked.append(r); seen.add(task); break
        rows=picked
    with out.open('w', newline='', encoding='utf-8') as f:
        fields=['batch_id','task','task_display','real_item_count','total_item_count','survey_url']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} rows to {out}')
if __name__ == '__main__': main()
