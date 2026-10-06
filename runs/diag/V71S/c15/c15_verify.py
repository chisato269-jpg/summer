import json, os, subprocess, sys, time
sp = os.path.dirname(os.path.abspath(__file__))
repo = os.getcwd()
def lines(n):
    p = os.path.join(sp, f"hb_{n}.txt")
    return len(open(p).read().splitlines()) if os.path.exists(p) else 0
def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out
ctrl = subprocess.Popen(f'cmd /c "python "{sp}\sleeper.py" ctrl & ping -n 60 127.0.0.1 >nul"', cwd=repo, shell=False)
det = subprocess.Popen(f'cmd /c "python -m rlkit.launch_detached --log "{sp}\det.log" --pidfile "{sp}\det_pid.json" -- python "{sp}\sleeper.py" det & ping -n 60 127.0.0.1 >nul"', cwd=repo, shell=False)
time.sleep(8)
print("before kill: ctrl lines", lines("ctrl"), "det lines", lines("det"))
info = json.load(open(os.path.join(sp, "det_pid.json")))
print("launch info: pid", info["pid"], "breakaway", info["breakaway"])
for p in (ctrl, det):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
time.sleep(4)
c1, d1 = lines("ctrl"), lines("det")
time.sleep(8)
c2, d2 = lines("ctrl"), lines("det")
print(f"after tree kill: ctrl lines {c1} -> {c2} ({'stopped' if c2 == c1 else 'still running'}); det lines {d1} -> {d2} ({'still running' if d2 > d1 else 'stopped'})")
print("det alive:", alive(info["pid"]))
subprocess.run(["taskkill", "/F", "/PID", str(info["pid"])], capture_output=True)
