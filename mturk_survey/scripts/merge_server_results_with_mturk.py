#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, csv, json, pathlib, re
CODE_RE = re.compile(r"SGV-[A-Za-z0-9_.-]+-[A-Z0-9]{8}")
BACKUP_PREFIX = "SGVJSON-"

def find_code(row: dict) -> str:
    keys = [k for k in row if k.lower().startswith('answer.')] + list(row.keys())
    seen=set()
    for k in keys:
        if k in seen: continue
        seen.add(k)
        val=str(row.get(k,'') or '').strip()
        if not val: continue
        if val.startswith(BACKUP_PREFIX): return val
        m=CODE_RE.search(val)
        if m: return m.group(0)
    return ''

def load_server_results(results_dir: pathlib.Path) -> dict:
    mapping={}
    for p in sorted(results_dir.glob('*.json')):
        try: obj=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        code=obj.get('server_code') or obj.get('client_code')
        if code: mapping[str(code)] = obj
    return mapping

def decode_backup(code: str) -> dict:
    raw=base64.b64decode(code[len(BACKUP_PREFIX):].encode('ascii'))
    return json.loads(raw.decode('utf-8'))

def first(row: dict, candidates: list[str]) -> str:
    for c in candidates:
        if c in row and row[c] != '': return row[c]
    return ''

def main() -> None:
    ap=argparse.ArgumentParser(description='Merge MTurk Survey Link codes with locally saved JSON answers.')
    ap.add_argument('--mturk-csv', required=True, help='Downloaded MTurk results CSV')
    ap.add_argument('--results-dir', default='results/server_saved', help='Directory with server-saved JSON files')
    ap.add_argument('--outdir', default='results/merged')
    args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    mturk_csv=pathlib.Path(args.mturk_csv); mturk_csv = mturk_csv if mturk_csv.is_absolute() else root/mturk_csv
    results_dir=pathlib.Path(args.results_dir); results_dir = results_dir if results_dir.is_absolute() else root/results_dir
    outdir=pathlib.Path(args.outdir); outdir = outdir if outdir.is_absolute() else root/outdir; outdir.mkdir(parents=True, exist_ok=True)
    server=load_server_results(results_dir)
    assignment_rows=[]; item_rows=[]
    with mturk_csv.open(newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        for row in reader:
            code=find_code(row); payload=None; source=''
            if code.startswith(BACKUP_PREFIX):
                try: payload=decode_backup(code); source='backup_code'
                except Exception as e: payload={'_decode_error':str(e)}; source='backup_decode_error'
            elif code in server:
                payload=server[code]; source='server_saved'
            else:
                source='missing_code' if code else 'no_code_found'
            assignment_id=first(row,['AssignmentId','assignmentId']); worker_id=first(row,['WorkerId','workerId']); hit_id=first(row,['HITId','hitId'])
            batch_id=payload.get('batch_id','') if isinstance(payload,dict) else ''; task=payload.get('task','') if isinstance(payload,dict) else ''
            assignment_rows.append({'AssignmentId':assignment_id,'WorkerId':worker_id,'HITId':hit_id,'completion_code':code,'code_source':source,'batch_id':batch_id,'task':task,'worker_id_entered':payload.get('worker_id_entered','') if isinstance(payload,dict) else '','server_saved_at':payload.get('server_saved_at','') if isinstance(payload,dict) else '','submitted_at':payload.get('submitted_at','') if isinstance(payload,dict) else ''})
            if isinstance(payload,dict) and isinstance(payload.get('responses'),list):
                for r in payload['responses']:
                    reasons=r.get('reasons',[])
                    item_rows.append({'AssignmentId':assignment_id,'WorkerId':worker_id,'HITId':hit_id,'completion_code':code,'batch_id':batch_id,'task':r.get('task',task),'item_id':r.get('item_id',''),'true_label':r.get('true_label',''),'label_preserved':r.get('label_preserved',''),'fluent':r.get('fluent',''),'reasons':';'.join(reasons if isinstance(reasons,list) else []),'note':r.get('note',''),'time_ms':r.get('time_ms','')})
    assign_out=outdir/'assignment_summary.csv'; item_out=outdir/'item_annotations.csv'
    with assign_out.open('w', newline='', encoding='utf-8') as f:
        fields=['AssignmentId','WorkerId','HITId','completion_code','code_source','batch_id','task','worker_id_entered','server_saved_at','submitted_at']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(assignment_rows)
    with item_out.open('w', newline='', encoding='utf-8') as f:
        fields=['AssignmentId','WorkerId','HITId','completion_code','batch_id','task','item_id','true_label','label_preserved','fluent','reasons','note','time_ms']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(item_rows)
    print(f'Wrote {len(assignment_rows)} assignments to {assign_out}')
    print(f'Wrote {len(item_rows)} item annotations to {item_out}')
if __name__ == '__main__': main()
