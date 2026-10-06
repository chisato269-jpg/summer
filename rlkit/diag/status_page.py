"""走行状況のページ（スマホの Claude アプリのコードタブなどで開ける HTML）を作る。

  python -m rlkit.diag.status_page            # runs/diag/V71S/status.html を書く

走行ディレクトリは読むだけ。ページには、生成時の状況を JSON で埋め込む。ページの中の JavaScript が現在時刻と最終更新の時刻を比べて、
生成から時間が経って古くなった場合や、更新が止まっている場合に、そのことを表示する（走行中のまま古い表示が残っても、
「稼働中」と言い続けない）。ページを更新するのは、このスクリプトを実行して Artifact として公開し直したとき。
"""
import datetime
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from rlkit import localdata

_B = localdata.latest_run()                    # OneDrive の外（~/rl_local/runs）と以前の走行（runs/）の両方から最新の走行を探す
RUN_DIR = _B[1] if _B else os.path.join(_REPO_ROOT, "runs", "v2s_trial08")
OUT = os.path.join(_REPO_ROOT, "runs", "diag", "V71S", "status.html")
PLANNED = _B[2] if _B else 1500
ALIVE_MIN = 20

TEMPLATE = r"""<title>学習の走行状況</title>
<style>
:root{--bg:#f6f7f5;--panel:#ffffff;--fg:#1d2420;--muted:#5b665f;--line:#dde2dc;--ok:#1a7f37;--ok-bg:#dcf3e2;--bad:#b3261e;--bad-bg:#fbe2df;--warn:#8a5a00;--warn-bg:#fdefc7;--ink:#2b5c8a;--ink2:#9aa7a0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121613;--panel:#1b211d;--fg:#e6ebe7;--muted:#9aa59e;--line:#2c352f;--ok:#58c47a;--ok-bg:#16301f;--bad:#ff8a80;--bad-bg:#3a1b19;--warn:#f0c15a;--warn-bg:#3a2e10;--ink:#7fb3e6;--ink2:#5d6a63;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121613;--panel:#1b211d;--fg:#e6ebe7;--muted:#9aa59e;--line:#2c352f;--ok:#58c47a;--ok-bg:#16301f;--bad:#ff8a80;--bad-bg:#3a1b19;--warn:#f0c15a;--warn-bg:#3a2e10;--ink:#7fb3e6;--ink2:#5d6a63;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif;padding-inline:16px;padding-block:16px 32px;max-width:720px;margin-inline:auto;line-height:1.5}
h1{font-size:1.05rem;margin:0 0 12px;font-weight:600;color:var(--muted);letter-spacing:.02em}
.state{border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:4px}
.state.ok{background:var(--ok-bg);color:var(--ok)}.state.bad{background:var(--bad-bg);color:var(--bad)}.state.warn{background:var(--warn-bg);color:var(--warn)}
.state .big{font-size:1.6rem;font-weight:700;line-height:1.2}
.state .sub{font-size:.9rem;color:var(--fg)}
.bar{height:10px;border-radius:5px;background:var(--line);overflow:hidden;margin-block:14px 4px}
.bar>i{display:block;height:100%;background:var(--ink)}
.prog{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted);font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-block:14px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0}
.cell b{display:block;font-size:1.15rem;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.cell span{font-size:.78rem;color:var(--muted)}
h2{font-size:.9rem;margin:20px 0 8px;color:var(--muted);font-weight:600}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
svg{display:block;width:100%;height:auto}
.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums;min-width:420px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
.note{font-size:.8rem;color:var(--muted);margin-top:16px}
</style>
<h1>夏季の学習の走行状況</h1>
<div id="state" class="state"><span class="big">読み込み中</span></div>
<div class="bar"><i id="barfill" style="width:0%"></i></div>
<div class="prog"><span id="p1"></span><span id="p2"></span></div>
<div class="grid" id="grid"></div>
<h2>収穫の ECI（直近 80 更新）</h2>
<div class="chart"><svg id="chart" viewBox="0 0 640 210" role="img" aria-label="収穫の ECI の推移"></svg></div>
<h2>直近 8 更新</h2>
<div class="tw"><table id="tbl"></table></div>
<p class="note" id="note"></p>
<script>
var D = __DATA__;
function pad(n){return (n<10?"0":"")+n}
function hm(ms){var d=new Date(ms);return pad(d.getMonth()+1)+"/"+pad(d.getDate())+" "+pad(d.getHours())+":"+pad(d.getMinutes())}
function ago(sec){var m=Math.floor(sec/60);if(m<1)return "1 分未満前";if(m<120)return m+" 分前";return (m/60).toFixed(1)+" 時間前"}
function render(){
  var now=Date.now(), lastAge=(now-D.last_update_ms)/1000, genAge=(now-D.generated_ms)/1000;
  var cls, big, sub;
  if(D.update>=D.planned){cls="ok";big="完了";sub="更新 "+D.planned+" に到達"}
  else if(D.stop_file){cls="bad";big="停止指示あり";sub="STOP ファイルがある"}
  else if(lastAge<=D.alive_s){cls="ok";big="稼働中";sub="最終更新 "+ago(lastAge)}
  else if(lastAge<=3600){cls="warn";big="更新が止まっている可能性";sub="最終更新 "+ago(lastAge)+"。この表示が古いだけの場合もある"}
  else{cls="bad";big="停止";sub="最終更新 "+ago(lastAge)+(D.stop_reason?"。"+D.stop_reason:"")}
  var el=document.getElementById("state");el.className="state "+cls;
  el.innerHTML='<span class="big">'+(cls==="ok"?"● ":cls==="warn"?"▲ ":"■ ")+big+'</span><span class="sub">'+sub+'</span>';
  document.getElementById("barfill").style.width=(100*D.update/D.planned).toFixed(1)+"%";
  document.getElementById("p1").textContent="更新 "+D.update+" / "+D.planned+"（"+(100*D.update/D.planned).toFixed(1)+"%）";
  document.getElementById("p2").textContent=D.eta_h!=null&&cls==="ok"?"残り約 "+(D.eta_h/24).toFixed(1)+" 日":"";
  var cells=[["最良の収穫の ECI",D.best_eci.toFixed(4),"更新 "+D.best_update],
    ["1 更新の実時間（直近 20 の中央値）",D.sec_median.toFixed(0)+" 秒",""],
    ["空きメモリの最小（直近 20 の最小）",D.min_free_gb==null?"記録なし":D.min_free_gb.toFixed(2)+" GB",""],
    ["一括読み取りの戻し（直近 20 の合計）",D.fallbacks==null?"記録なし":String(D.fallbacks),"回"],
    ["再開からの更新数",String(D.updates_since_resume),"V7.1-S は更新 685 から"],
    ["最終更新の時刻",hm(D.last_update_ms),""]];
  document.getElementById("grid").innerHTML=cells.map(function(c){return '<div class="cell"><span>'+c[0]+'</span><b>'+c[1]+'</b><span>'+c[2]+'</span></div>'}).join("");
  var note="この表示の作成: "+hm(D.generated_ms)+"（"+ago(genAge)+"）。ページは作り直したときだけ更新される。";
  if(genAge>2700) note+=" 作成から 45 分以上経っているので、上の状態は現在と違う可能性がある。";
  document.getElementById("note").textContent=note;
}
function chart(){
  var s=D.series, W=640,H=210,L=44,R=10,T=10,B=24;
  var vals=s.map(function(r){return r[1]}).concat(s.map(function(r){return r[2]}));
  var lo=Math.min.apply(null,vals), hi=Math.max.apply(null,vals), pad=(hi-lo)*0.1||0.01; lo-=pad; hi+=pad;
  var n=s.length, x=function(i){return L+(W-L-R)*i/Math.max(n-1,1)}, y=function(v){return T+(H-T-B)*(1-(v-lo)/(hi-lo))};
  var css=getComputedStyle(document.documentElement), line=css.getPropertyValue("--line"), muted=css.getPropertyValue("--muted"), ink=css.getPropertyValue("--ink"), ink2=css.getPropertyValue("--ink2");
  var g="";
  for(var k=0;k<=4;k++){var v=lo+(hi-lo)*k/4;g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="'+line+'" stroke-width="1"/><text x="'+(L-6)+'" y="'+(y(v)+4)+'" text-anchor="end" font-size="11" fill="'+muted+'">'+v.toFixed(3)+'</text>'}
  var p1=s.map(function(r,i){return (i?"L":"M")+x(i).toFixed(1)+" "+y(r[1]).toFixed(1)}).join(" ");
  var p2=s.map(function(r,i){return (i?"L":"M")+x(i).toFixed(1)+" "+y(r[2]).toFixed(1)}).join(" ");
  var by=D.best_eci; g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(by)+'" y2="'+y(by)+'" stroke="'+ink+'" stroke-width="1" stroke-dasharray="4 4" opacity=".6"/>';
  g+='<path d="'+p2+'" fill="none" stroke="'+ink2+'" stroke-width="1.4"/><path d="'+p1+'" fill="none" stroke="'+ink+'" stroke-width="2"/>';
  g+='<circle cx="'+x(n-1)+'" cy="'+y(s[n-1][1])+'" r="3.5" fill="'+ink+'"/>';
  g+='<text x="'+L+'" y="'+(H-6)+'" font-size="11" fill="'+muted+'">更新 '+s[0][0]+'</text><text x="'+(W-R)+'" y="'+(H-6)+'" text-anchor="end" font-size="11" fill="'+muted+'">更新 '+s[n-1][0]+'　（濃い線: 収穫の最良、薄い線: 決定論的評価、破線: これまでの最良）</text>';
  document.getElementById("chart").innerHTML=g;
}
function table(){
  var h="<tr><th>更新</th><th>実時間 [秒]</th><th>収穫 ECI</th><th>評価 ECI</th><th>戻し</th><th>空き [GB]</th></tr>";
  D.recent.forEach(function(r){h+="<tr><td>"+r[0]+"</td><td>"+r[1].toFixed(0)+"</td><td>"+r[2].toFixed(4)+"</td><td>"+r[3].toFixed(4)+"</td><td>"+(r[4]==null?"-":r[4])+"</td><td>"+(r[5]==null?"-":r[5].toFixed(2))+"</td></tr>"});
  document.getElementById("tbl").innerHTML=h;
}
render();chart();table();setInterval(render,30000);
</script>
"""


def main():
    ups = []
    with open(os.path.join(RUN_DIR, "updates.jsonl"), encoding="utf-8") as f:
        lines = f.readlines()
    for l in lines[-200:]:
        ups.append(json.loads(l))
    last = ups[-1]
    mtime = os.path.getmtime(os.path.join(RUN_DIR, "updates.jsonl"))
    recent20 = ups[-20:]
    secs = sorted(u["update_seconds"] for u in recent20)
    med = secs[len(secs) // 2]
    c13 = [u.get("c13") for u in recent20 if u.get("c13")]
    since = [u for u in ups if u["update"] > 684]
    stop_reason = None
    try:
        st = json.load(open(os.path.join(RUN_DIR, "status.json"), encoding="utf-8"))
        if st.get("update_index") == last["update"] and st.get("stop_reason"):
            stop_reason = ("例外で停止した" if st.get("status") == "error" else str(st["stop_reason"])[:60])
    except Exception:
        pass
    best = last["harvest_best"]
    data = {
        "generated_ms": int(datetime.datetime.now().timestamp() * 1000), "last_update_ms": int(mtime * 1000),
        "update": last["update"], "planned": PLANNED, "alive_s": ALIVE_MIN * 60,
        "stop_file": os.path.exists(os.path.join(RUN_DIR, "STOP")), "stop_reason": stop_reason,
        "best_eci": best["ECI"], "best_update": best["update"], "sec_median": med,
        "eta_h": (PLANNED - last["update"]) * med / 3600.0,
        "min_free_gb": (min(c["min_free_memory_gb"] for c in c13) if c13 else None),
        "fallbacks": (sum(c["rpm_fallback_chunks"] for c in c13) if c13 else None),
        "updates_since_resume": len(since),
        "series": [[u["update"], u["harvest_update_ECI"], u["eval_ECI"]] for u in ups[-80:]],
        "recent": [[u["update"], u["update_seconds"], u["harvest_update_ECI"], u["eval_ECI"],
                    (u.get("c13") or {}).get("rpm_fallback_chunks"), (u.get("c13") or {}).get("min_free_memory_gb")]
                   for u in ups[-8:]],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(OUT, last["update"], "alive" if (datetime.datetime.now().timestamp() - mtime) <= ALIVE_MIN * 60 else "not alive")


if __name__ == "__main__":
    main()
