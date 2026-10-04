#!/usr/bin/env python3
"""results.jsonl と outputs/ から聴き比べ用 compare.html を生成する。"""
import html
import json
from pathlib import Path

from texts import TEXTS

BENCH = Path(__file__).resolve().parent

rows = [json.loads(l) for l in (BENCH / "results.jsonl").open() if '"error"' not in l]
cfgs = list(dict.fromkeys(r["config"] for r in rows))
d = {(r["config"], r["text_id"]): r for r in rows}

CSS = """
:root{--bg:#fff;--fg:#17171a;--mut:#6b6b73;--line:#e4e4e8;--card:#f7f7f9;--acc:#2f6df6;--accfg:#fff}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--fg:#ececf0;--mut:#9b9ba3;--line:#2b2b31;--card:#1b1b1f;--acc:#6f9bff;--accfg:#10131c}}
:root[data-theme="dark"]{--bg:#121214;--fg:#ececf0;--mut:#9b9ba3;--line:#2b2b31;--card:#1b1b1f;--acc:#6f9bff;--accfg:#10131c}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:28px 16px 64px;line-height:1.6;
 font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:1.4rem;margin:0 0 .3rem}
.sub{color:var(--mut);font-size:.88rem;margin-bottom:2rem}
section{margin-bottom:2.5rem;border-top:1px solid var(--line);padding-top:1.2rem}
h2{font-size:1rem;margin:0 0 .5rem;color:var(--mut);font-weight:600}
.txt{background:var(--card);padding:10px 13px;border-radius:9px;font-size:.93rem;margin:0 0 .9rem}
audio{width:100%;height:38px;display:block;margin-bottom:.7rem}
.btns{display:flex;flex-wrap:wrap;gap:6px}
button{font:inherit;font-size:.8rem;padding:5px 11px;border-radius:999px;cursor:pointer;
 border:1px solid var(--line);background:var(--card);color:var(--fg);font-variant-numeric:tabular-nums}
button:hover{border-color:var(--acc)}
button[aria-pressed="true"]{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
button .t{opacity:.65;margin-left:6px}
.hint{color:var(--mut);font-size:.8rem;margin-top:.6rem}
kbd{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:.78rem}
@media(max-width:560px){audio{height:34px}}
"""

JS = """
document.querySelectorAll('section[data-text]').forEach(sec=>{
  const player=sec.querySelector('audio');
  const btns=[...sec.querySelectorAll('button[data-src]')];
  const pick=(btn)=>{
    const t=player.currentTime, playing=!player.paused;
    btns.forEach(b=>b.setAttribute('aria-pressed', String(b===btn)));
    player.src=btn.dataset.src;
    // 再生位置を維持したまま差し替えて A/B する
    player.addEventListener('loadedmetadata',()=>{
      player.currentTime=Math.min(t, Math.max(0,player.duration-0.05));
      if(playing) player.play();
    },{once:true});
    player.load();
  };
  btns.forEach(b=>b.addEventListener('click',()=>pick(b)));
  sec.addEventListener('keydown',e=>{
    const i=parseInt(e.key,10);
    if(i>=1&&i<=btns.length){ pick(btns[i-1]); e.preventDefault(); }
  });
  sec.tabIndex=0;
});
"""

p = [
    '<!doctype html><html lang="ja"><head><meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width,initial-scale=1">',
    "<title>Irodori-TTS 聴き比べ</title>",
    f"<style>{CSS}</style></head><body><div class='wrap'>",
    "<h1>Irodori-TTS v4 聴き比べ</h1>",
    "<p class='sub'>RTX 5090 / RF 40 steps / seed 0 / リファレンス音声なし。"
    "括弧内は生成時間。ボタンで再生位置を保ったままモデルを切り替えられる。</p>",
]
for tid, text in TEXTS:
    p.append(f"<section data-text='{tid}'><h2>{tid}</h2>")
    p.append(f"<p class='txt'>{html.escape(text)}</p>")
    first = next((c for c in cfgs if (c, tid) in d), None)
    if first is None:
        p.append("</section>")
        continue
    p.append(f"<audio controls preload='metadata' src='outputs/{tid}/{first}.wav'></audio>")
    p.append("<div class='btns'>")
    for i, c in enumerate(cfgs):
        r = d.get((c, tid))
        if not r:
            continue
        pressed = "true" if c == first else "false"
        p.append(
            f"<button data-src='outputs/{tid}/{c}.wav' aria-pressed='{pressed}'>"
            f"{i + 1}. {c}<span class='t'>{r['gen_sec_best']:.2f}s</span></button>"
        )
    p.append("</div>")
    p.append("<p class='hint'>このブロックをクリックしてから <kbd>1</kbd>〜<kbd>9</kbd> でも切り替え可</p>")
    p.append("</section>")
p.append(f"</div><script>{JS}</script></body></html>")

out = BENCH / "compare.html"
out.write_text("\n".join(p), encoding="utf-8")
print(f"wrote {out}")
