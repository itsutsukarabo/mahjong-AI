"""v44 monitor3: ep36 -> early_stop (resumed after session break at ep35)
- 安全弁: train_nll 2epoch連続上昇 → 停止報告
- ep100: 報告のみ（停止しない）
- プロセス終了（early stopping or crash）: 最終報告
"""
import re, time, os, subprocess, json

LOG  = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_resume4_stdout.log"
CKPT = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44\checkpoint.pt"
TLOG = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44\train_log.json"
PIDF = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_training3.pid"
OUT  = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_monitor3_output2.log"

epoch_re = re.compile(r"^epoch\s+(\d+)\s+nll=([\d.]+)")

_fh = open(OUT, "w", encoding="utf-8", buffering=1)

def p(msg):
    print(msg, flush=True)
    _fh.write(msg + "\n"); _fh.flush()

def get_ckpt_mtime():
    try: return os.path.getmtime(CKPT)
    except: return 0

def kill_pid(pid):
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    p(f"[monitor] Killed PID {pid}")

def is_alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                       capture_output=True, text=True)
    return str(pid) in r.stdout

def wait_for_ckpt(pid, ref, timeout=720):
    p("[monitor] Waiting for checkpoint save...")
    for i in range(timeout // 10):
        time.sleep(10)
        if get_ckpt_mtime() > ref + 0.5:
            p(f"[monitor] Checkpoint saved (~{(i+1)*10}s)")
            return True
        if not is_alive(pid):
            p("[monitor] Process ended during wait")
            return False
    p("[monitor] Checkpoint wait timed out")
    return False

def read_tlog():
    try:
        return [json.loads(l) for l in open(TLOG, encoding="utf-8").read().splitlines() if l.strip()]
    except:
        return []

def metrics_for(entries, ep):
    e = next((x for x in entries if x["epoch"] == ep), None)
    if e:
        p(f"  val_eae={e.get('val_eae')}  val_eae_stage={e.get('val_eae_stage')}")
        p(f"  val_acc={e.get('val_acc')}  shanten_acc={e.get('shanten_acc')}")
        p(f"  wait_f1={e.get('wait_f1')}  composite={e.get('composite')}")
    else:
        p(f"  ep{ep} not found in train_log")

def final_report(nll_history):
    p("[monitor] === FINAL REPORT ===")
    entries = read_tlog()
    if entries:
        best = min(entries, key=lambda e: e.get("val_eae", float("inf")))
        p(f"Best epoch: ep{best['epoch']}")
        p(f"  val_eae={best.get('val_eae')}  val_eae_stage={best.get('val_eae_stage')}")
        p(f"  val_acc={best.get('val_acc')}  shanten_acc={best.get('shanten_acc')}")
        p(f"  wait_f1={best.get('wait_f1')}  composite={best.get('composite')}")
        last = entries[-1]
        p(f"Last epoch: ep{last['epoch']}")
        p(f"  val_eae={last.get('val_eae')}  val_eae_stage={last.get('val_eae_stage')}")
        p(f"  val_acc={last.get('val_acc')}")
    if nll_history:
        last_ep, last_nll = nll_history[-1]
        p(f"Final train_nll: ep{last_ep} nll={last_nll:.4f}")
        p(f"  v39 ep200=0.0591  v43 ep68=0.5056  v44 ep35=0.3351")
    p("[monitor] === END ===")

def scan_log(path, seen, nll_history):
    new = []
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            m = epoch_re.match(line.strip())
            if not m: continue
            ep, nll = int(m.group(1)), float(m.group(2))
            if ep in seen: continue
            seen.add(ep)
            nll_history.append((ep, nll))
            new.append((ep, nll))
    except:
        pass
    return new

def main():
    p("[monitor] Monitor3 started (ep36 -> early_stop)")

    for _ in range(60):
        if os.path.exists(PIDF):
            pid = int(open(PIDF).read().strip())
            p(f"[monitor] Training PID={pid}")
            break
        time.sleep(2)
    else:
        p("[monitor] ERROR: PID file not found after 120s")
        return

    seen = set()
    nll_history = []
    ep100_reported = False

    p("[monitor] Watching (safety: 2ep nll rise / ep100 report / early_stop final)")

    while True:
        time.sleep(20)

        alive = is_alive(pid)
        new_eps = scan_log(LOG, seen, nll_history)

        for ep, nll in new_eps:
            p(f"[monitor] ep{ep:3d}  nll={nll:.4f}")

            # 安全弁: 2 epoch 連続 nll 上昇
            if len(nll_history) >= 3:
                _, n2 = nll_history[-1]
                _, n1 = nll_history[-2]
                _, n0 = nll_history[-3]
                if n2 > n1 and n1 > n0:
                    e0 = nll_history[-3][0]; e1 = nll_history[-2][0]
                    p(f"[monitor] !!! NLL_REGRESSION ep{e0}:{n0:.4f} -> ep{e1}:{n1:.4f} -> ep{ep}:{n2:.4f}")
                    ref = get_ckpt_mtime()
                    wait_for_ckpt(pid, ref)
                    kill_pid(pid)
                    entries = read_tlog()
                    p(f"[monitor] ep{ep} metrics:")
                    metrics_for(entries, ep)
                    final_report(nll_history)
                    _fh.close(); return

            # ep100: 報告のみ・継続
            if ep == 100 and not ep100_reported:
                ep100_reported = True
                p("[monitor] === EP100 REPORT ===")
                entries = read_tlog()
                metrics_for(entries, 100)
                p(f"  train_nll={nll:.4f}  (v39 ep100 est≈0.176  v43 ep68=0.5056)")
                p("[monitor] Continuing (no stop)...")

        # プロセス終了 = early stopping 発動 or crash
        if not alive:
            for ep, nll in scan_log(LOG, seen, nll_history):
                p(f"[monitor] ep{ep:3d}  nll={nll:.4f}")
            p("[monitor] Training process ended")
            final_report(nll_history)
            _fh.close(); return

if __name__ == "__main__":
    main()
