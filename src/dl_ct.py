import os, sys, time, threading, urllib.request, json

URL = "https://zenodo.org/api/records/3757476/files/COVID-19-CT-Seg_20cases.zip/content"
OUT = "/Users/a1/.openclaw/workspace/isbi-paper/data/raw/CT.zip"
PROXY = "http://127.0.0.1:15236"
TOTAL = 1105395141
N = 16

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if not os.path.exists(OUT):
    with open(OUT, "wb") as f:
        f.truncate(TOTAL)

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
)

lock = threading.Lock()
done = [0] * N
chunk = TOTAL // N

def worker(i):
    start = i * chunk
    end = TOTAL - 1 if i == N - 1 else (start + chunk - 1)
    for attempt in range(6):
        have = done[i]
        s = start + have
        if s > end:
            return
        req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{end}", "User-Agent": "Mozilla/5.0"})
        try:
            with opener.open(req, timeout=120) as r, open(OUT, "r+b") as f:
                f.seek(s)
                while True:
                    buf = r.read(262144)
                    if not buf:
                        break
                    f.write(buf)
                    with lock:
                        done[i] += len(buf)
        except Exception:
            time.sleep(2 + attempt * 2)
            continue
        if done[i] >= end - start + 1:
            return

threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
for t in threads: t.start()

t0 = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(10)
    got = sum(done)
    pct = got / TOTAL * 100
    rate = got / max(1, time.time() - t0)
    sys.stdout.write(f"\r{pct:5.1f}%  {got/1e6:.1f}MB / {TOTAL/1e6:.0f}MB  {rate/1024:.0f}KB/s   ")
    sys.stdout.flush()
print("\ndone")
