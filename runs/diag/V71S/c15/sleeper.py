import sys, time, os
name = sys.argv[1]; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hb_{name}.txt")
for i in range(90):
    open(out, "a").write(f"{time.time():.0f} pid={os.getpid()}\n")
    time.sleep(2)
