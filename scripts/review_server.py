"""A local page for the hand review of real-audio questions.

Serves one row of a review CSV at a time: the clip plays, the question is shown
with its answer windows drawn on a timeline, and one keystroke records the
verdict straight back into the CSV (the file ctag.real_data apply-review reads).

    python scripts/review_server.py --csv data/desed/public/review_sample.csv
    # then open http://127.0.0.1:8765  (opens by itself)

Keys: space = play the whole clip, a = play the answer windows only, y = answer is
right, n = question is unfair or unanswerable, c = type a corrected answer,
left/right = previous/next row, u = next unreviewed row. Every change is saved
at once; close the tab whenever you like and come back.

Nothing leaves the machine: the server binds to 127.0.0.1 and serves only the
CSV and the audio files it names, from inside the project directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["qid", "clip_id", "audio", "qtype", "text", "answer", "expects_empty", "ok", "corrected_answer", "note"]


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys()) if rows else COLUMNS
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)                      # never a half-written review file


def apply_verdict(rows: list[dict], qid: str, ok: str, corrected: str, note: str) -> dict:
    """Update one row in place. ok is 'y', 'n' or '' (cleared); a corrected
    answer must parse as a JSON list of [start, end] pairs or be empty."""
    if ok not in ("y", "n", ""):
        raise ValueError(f"ok must be y, n or empty, got {ok!r}")
    corrected = corrected.strip()
    if corrected:
        try:
            iv = json.loads(corrected)
        except json.JSONDecodeError:
            iv = None
        if not isinstance(iv, list) or not all(isinstance(p, list) and len(p) == 2 and p[0] <= p[1] for p in iv):
            raise ValueError("corrected_answer must be a JSON list of [start, end] pairs, e.g. [[1.2, 2.0]]")
        corrected = json.dumps([[round(float(a), 3), round(float(b), 3)] for a, b in iv])
    for r in rows:
        if r["qid"] == qid:
            r["ok"], r["corrected_answer"], r["note"] = ok, corrected, note.strip()
            return r
    raise KeyError(qid)


def progress(rows: list[dict]) -> dict:
    done = [r for r in rows if (r.get("ok") or "").strip()]
    return {"total": len(rows), "done": len(done), "yes": sum(1 for r in done if r["ok"] == "y"),
            "no": sum(1 for r in done if r["ok"] == "n"), "corrected": sum(1 for r in done if (r.get("corrected_answer") or "").strip())}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hand review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--fg:#1b1f24;--muted:#5c6470;--card:#fff;--line:#d9dde3;--acc:#2456d6;--ok:#1f8a4c;--no:#c0392b;--gold:#f0b429;--sel:#dbe6ff}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e8eaed;--muted:#9aa3ad;--card:#1d2127;--line:#2e343c;--acc:#7aa2ff;--ok:#4cc38a;--no:#ff7b72;--gold:#ffcc55;--sel:#263654}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0}
header b{font-size:16px}.bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
main{display:grid;grid-template-columns:260px 1fr;gap:16px;padding:16px;max-width:1200px;margin:0 auto}
@media(max-width:800px){main{grid-template-columns:1fr}aside{max-height:160px}}
aside{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:auto;max-height:calc(100vh - 100px)}
aside div{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px;font-size:13px}aside div.sel{background:var(--sel)}
aside span.t{color:var(--muted);width:92px;flex:none}aside span.v{width:14px;flex:none;font-weight:700}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.q{font-size:20px;margin:6px 0 10px}.tag{display:inline-block;font-size:12px;padding:2px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);margin-right:8px}
canvas{width:100%;height:84px;display:block;margin:10px 0;border-radius:6px;background:var(--bg)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.y{border-color:var(--ok);color:var(--ok)}button.n{border-color:var(--no);color:var(--no)}button.on{background:var(--sel)}
kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--muted)}
input,textarea{font:inherit;width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.muted{color:var(--muted);font-size:13px}.err{color:var(--no)}
</style></head><body>
<header><b>Hand review</b><span id="prog" class="muted"></span><div class="bar"><i id="fill" style="width:0"></i></div><span class="muted" id="file"></span></header>
<main><aside id="list"></aside>
<div class="card">
<div><span class="tag" id="type"></span><span class="muted" id="clip"></span></div>
<div class="q" id="text"></div>
<audio id="au" controls preload="auto" style="width:100%"></audio>
<canvas id="cv" height="84"></canvas>
<div class="muted">Gold answer: <code id="ans"></code> <span id="empty"></span></div>
<div class="row"><button id="play">Play clip <kbd>space</kbd></button><button id="playw">Play answer windows <kbd>a</kbd></button>
<button class="y" id="by">Right <kbd>y</kbd></button><button class="n" id="bn">Unfair / unanswerable <kbd>n</kbd></button>
<button id="prev">&larr; prev</button><button id="next">next &rarr;</button><button id="unrev">next unreviewed <kbd>u</kbd></button></div>
<div class="row"><div style="flex:1;min-width:240px"><div class="muted">Corrected answer, only if the timing is off (JSON pairs, e.g. [[1.2, 2.0]]) <kbd>c</kbd></div><input id="corr" placeholder="leave empty if the gold answer is right"></div></div>
<div class="row"><div style="flex:1"><div class="muted">Note (optional)</div><input id="note" placeholder="one phrase, e.g. second dog bark not annotated"></div></div>
<div class="muted" id="status"></div>
</div></main>
<script>
let rows=[],i=0,stopAt=null,winIdx=0,wins=[];
const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/rows');const d=await r.json();rows=d.rows;$('file').textContent=d.file;renderList();const u=rows.findIndex(x=>!x.ok);show(u<0?0:u);}
function renderList(){const L=$('list');L.innerHTML='';rows.forEach((r,k)=>{const d=document.createElement('div');d.className=k===i?'sel':'';d.innerHTML=`<span class="v" style="color:${r.ok==='y'?'var(--ok)':r.ok==='n'?'var(--no)':'var(--muted)'}">${r.ok||'·'}</span><span class="t">${r.qtype}</span><span>${k+1}</span>`;d.onclick=()=>show(k);L.appendChild(d);});}
function prog(){const d=rows.filter(r=>r.ok).length;$('prog').textContent=`${d} / ${rows.length} reviewed`;$('fill').style.width=(100*d/rows.length)+'%';}
function parseIv(s){try{const v=JSON.parse(s||'[]');return Array.isArray(v)?v:[]}catch(e){return[]}}
function show(k){i=Math.max(0,Math.min(rows.length-1,k));const r=rows[i];$('type').textContent=r.qtype;$('clip').textContent=r.clip_id;$('text').textContent=r.text;
$('ans').textContent=r.answer;$('empty').textContent=r.expects_empty==='True'?'(the correct answer is empty: the sound should not be there)':'';
$('corr').value=r.corrected_answer||'';$('note').value=r.note||'';$('by').className='y'+(r.ok==='y'?' on':'');$('bn').className='n'+(r.ok==='n'?' on':'');
const au=$('au');au.src='/audio/'+encodeURIComponent(r.audio);au.load();stopAt=null;wins=parseIv(r.corrected_answer||r.answer);
[...$('list').children].forEach((d,k)=>d.className=k===i?'sel':'');const sel=$('list').children[i];if(sel)sel.scrollIntoView({block:'nearest'});prog();draw();$('status').textContent='';}
function draw(){const cv=$('cv'),ctx=cv.getContext('2d');cv.width=cv.clientWidth*devicePixelRatio;cv.height=84*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);const W=cv.clientWidth,H=84;const dur=$('au').duration||10;
ctx.clearRect(0,0,W,H);ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--line');for(let s=0;s<=dur;s++){const x=W*s/dur;ctx.fillRect(x,H-14,1,6);}
ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--muted');ctx.font='11px sans-serif';for(let s=0;s<=dur;s+=2){ctx.fillText(s+'s',W*s/dur+2,H-1);}
ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--gold');for(const [a,b] of wins){ctx.fillRect(W*a/dur,18,Math.max(2,W*(b-a)/dur),36);}
const t=$('au').currentTime||0;ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--acc');ctx.fillRect(W*t/dur-1,8,2,H-22);}
$('au').addEventListener('timeupdate',()=>{draw();if(stopAt!==null&&$('au').currentTime>=stopAt){winIdx++;if(winIdx<wins.length){$('au').currentTime=wins[winIdx][0];stopAt=wins[winIdx][1];}else{$('au').pause();stopAt=null;}}});
$('au').addEventListener('loadedmetadata',draw);
function playAll(){stopAt=null;const au=$('au');au.currentTime=0;au.play();}
function playWins(){if(!wins.length){$('status').textContent='The gold answer is empty: nothing to play. Listen to the whole clip for the sound instead.';return;}winIdx=0;const au=$('au');au.currentTime=Math.max(0,wins[0][0]-0.3);stopAt=wins[0][1]+0.3;au.play();}
async function save(ok){const r=rows[i];const body={qid:r.qid,ok:ok===undefined?r.ok:ok,corrected_answer:$('corr').value,note:$('note').value};
const res=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await res.json();
if(d.error){$('status').innerHTML='<span class="err">'+d.error+'</span>';return false;}Object.assign(r,d.row);$('by').className='y'+(r.ok==='y'?' on':'');$('bn').className='n'+(r.ok==='n'?' on':'');renderList();prog();$('status').textContent='saved';return true;}
async function verdict(v){if(await save(v)){setTimeout(()=>nextUnreviewed(),250);}}
function nextUnreviewed(){let k=rows.findIndex((r,j)=>j>i&&!r.ok);if(k<0)k=rows.findIndex(r=>!r.ok);if(k<0){$('status').textContent='All rows reviewed. Run apply-review (docs/real_recordings.md).';return;}show(k);}
$('play').onclick=playAll;$('playw').onclick=playWins;$('by').onclick=()=>verdict('y');$('bn').onclick=()=>verdict('n');
$('prev').onclick=()=>show(i-1);$('next').onclick=()=>show(i+1);$('unrev').onclick=nextUnreviewed;
$('corr').addEventListener('change',()=>save());$('note').addEventListener('change',()=>save());
document.addEventListener('keydown',e=>{const typing=['INPUT','TEXTAREA'].includes(document.activeElement.tagName);if(typing){if(e.key==='Escape')document.activeElement.blur();return;}
if(e.code==='Space'){e.preventDefault();playAll();}else if(e.key==='a')playWins();else if(e.key==='y')verdict('y');else if(e.key==='n')verdict('n');
else if(e.key==='c'){$('corr').focus();}else if(e.key==='ArrowLeft')show(i-1);else if(e.key==='ArrowRight')show(i+1);else if(e.key==='u')nextUnreviewed();});
window.addEventListener('resize',draw);load();
</script></body></html>"""


def make_handler(csv_path: Path):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/rows":
                with lock:
                    rows = read_rows(csv_path)
                return self._send(200, {"rows": rows, "file": str(csv_path), "progress": progress(rows)})
            if u.path.startswith("/audio/"):
                rel = unquote(u.path[len("/audio/"):])
                p = (ROOT / rel).resolve()
                if not str(p).startswith(str(ROOT)) or not p.is_file():
                    return self._send(404, {"error": "no such audio"})
                data = p.read_bytes()
                return self._send(200, data, "audio/wav" if p.suffix == ".wav" else "application/octet-stream")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                with lock:
                    rows = read_rows(csv_path)
                    row = apply_verdict(rows, body["qid"], body.get("ok") or "", body.get("corrected_answer") or "",
                                        body.get("note") or "")
                    write_rows(csv_path, rows)
                return self._send(200, {"row": row, "progress": progress(rows)})
            except (ValueError, KeyError) as e:
                return self._send(400, {"error": str(e)})

        def log_message(self, *a):            # quiet
            pass

    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/desed/public/review_sample.csv")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found (python -m ctag.real_data export-review ...)")
    rows = read_rows(csv_path)
    print(f"{len(rows)} rows in {csv_path}; {progress(rows)['done']} reviewed")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(csv_path))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"review page: {url}  (Ctrl-C to stop; every verdict is saved at once)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
