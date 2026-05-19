#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, json, pathlib, random, re, string
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
ROOT = pathlib.Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
RESULTS = ROOT / "results" / "server_saved"
RESULTS.mkdir(parents=True, exist_ok=True)
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

def local_ip() -> str:
    return "YOUR_LAN_IP"

def make_code(batch_id: str) -> str:
    rand="".join(random.choice(string.ascii_uppercase+string.digits) for _ in range(8))
    return f"SGV-{SAFE.sub('_', batch_id or 'batch')[:32]}-{rand}"

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(PUBLIC),**kwargs)
    def _send_json(self,status:int,obj:dict)->None:
        data=json.dumps(obj,ensure_ascii=False).encode('utf-8')
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path=='/health': self._send_json(200,{'ok':True}); return
        if parsed.path=='/': self.send_response(302); self.send_header('Location','/index.html'); self.end_headers(); return
        return super().do_GET()
    def do_POST(self):
        parsed=urlparse(self.path)
        if parsed.path!='/api/submit': self._send_json(404,{'ok':False,'error':'unknown endpoint'}); return
        try:
            length=int(self.headers.get('Content-Length','0')); raw=self.rfile.read(length); payload=json.loads(raw.decode('utf-8'))
            batch_id=str(payload.get('batch_id','batch')); code=make_code(batch_id)
            payload['server_code']=code; payload['server_saved_at']=dt.datetime.utcnow().isoformat(timespec='seconds')+'Z'
            ts=dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f'); safe_batch=SAFE.sub('_',batch_id)[:80]
            path=RESULTS / f"{ts}_{safe_batch}_{code}.json"
            path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            self._send_json(200,{'ok':True,'code':code})
        except Exception as e:
            self._send_json(500,{'ok':False,'error':str(e)})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=8000); args=ap.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler); ip=local_ip()
    print('\nSemantic Gate Survey Link server is running.\n')
    print(f'Open locally:       http://localhost:{args.port}/index.html')
    print(f'LAN link:           http://{ip}:{args.port}/index.html')
    print(f'Results save to:    {RESULTS}')
    print('\nFor remote MTurk workers, expose this with Cloudflare Tunnel:')
    print(f'  cloudflared tunnel --url http://localhost:{args.port}')
    print('\nKeep this window open until workers are done.\n')
    try: server.serve_forever()
    except KeyboardInterrupt: print('\nStopped.')
if __name__=='__main__': main()
