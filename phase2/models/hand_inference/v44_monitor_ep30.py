"""v44 ep13->ep30 monitor: ep30到達で checkpoint 保存確認後に学習を停止してレポート。"""
import re, time, os, subprocess, json, sys

LOG   = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_resume_stdout.log"
CKPT  = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44\checkpoint.pt"
TLOG  = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44\train_log.json"
PIDF  = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_training.pid"
OUT   = r"C:\Users\Administrator.THPC-L01\mahjong-AI\phase2\models\hand_inference\v44_monitor_output.log"

STOP_AT = 30

epoch_re = re.compile(r"^epoch\s+(\d+)\s+nll=([\d.]+)")

_out_fh = open(OUT, "w", encoding="utf-8", buffering=1)

def p(msg):
    print(msg, flush=True)
    _out_fh.write(msg + "\n"); _out_fh.flush()

def get_ckpt_mtime():
    try: return os.path.getmtime(CKPT)
    except: return 0

def kill_pid(pid):
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    p(f"[monitor] Killed PID {pid}")

def wait_for_ckpt(pid, ref_mtime, timeout=720):
    p(f"[monitor] Waiting for checkpoint save (ref_mtime={ref_mtime:.1f})...")
    for i in range(timeout // 10):
        time.sleep(10)
        cur = get_ckpt_mtime()
        if cur > ref_mtime + 0.5:
            p(f"[monitor] Checkpoint confirmed saved (~{(i+1)*10}s elapsed)")
            return True
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True)
        if str(pid) not in r.stdout:
            p("[monitor] Training process ended during wait")
            return False
    p("[monitor] Timeout waiting for checkpoint (720s)")
    return False

def main():
    p("[monitor] Starting ep30 monitor...")
    # Wait for PID file
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

    p(f"[monitor] Monitoring ep13+ until ep{STOP_AT}...")

    while True:
        time.sleep(20)

        # Check training alive
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True)
        if str(pid) not in r.stdout:
            p("[monitor] Training process ended unexpectedly — summary:")
            for e, n in nll_history:
                p(f"  ep{e:3d}: nll={n:.4f}")
            break

        # Parse stdout log for new epoch lines
        try:
            lines = open(LOG, encoding="utf-8", errors="replace").readlines()
        except Exception as ex:
            p(f"[monitor] Log read error: {ex}")
            continue

        for line in lines:
            m = epoch_re.match(line.strip())
            if not m: continue
            ep, nll = int(m.group(1)), float(m.group(2))
            if ep in seen: continue
            seen.add(ep)
            nll_history.append((ep, nll))
            p(f"[monitor] ep{ep:3d}  nll={nll:.4f}")

            if ep >= STOP_AT:
                p(f"[monitor] === EP{STOP_AT} REACHED ===")
                ref = get_ckpt_mtime()
                wait_for_ckpt(pid, ref)
                kill_pid(pid)

                # Print nll history
                p("[monitor] === TRAIN NLL HISTORY (resumed run) ===")
                for e, n in nll_history:
                    p(f"  ep{e:3d}: nll={n:.4f}")

                # Print train_log metrics for ep30
                try:
                    entries = [json.loads(l) for l in
                               open(TLOG, encoding="utf-8").read().splitlines() if l.strip()]
                    for target_ep in [30]:
                        entry = next((e for e in entries if e["epoch"] == target_ep), None)
                        if entry:
                            p(f"[monitor] ep{target_ep} metrics:")
                            p(f"  val_eae={entry.get('val_eae')}  val_eae_stage={entry.get('val_eae_stage')}")
                            p(f"  val_acc={entry.get('val_acc')}  shanten_acc={entry.get('shanten_acc')}")
                            p(f"  wait_f1={entry.get('wait_f1')}  composite={entry.get('composite')}")
                except Exception as ex:
                    p(f"[monitor] train_log read error: {ex}")

                p("[monitor] === DONE ===")
                _out_fh.close()
                return

if __name__ == "__main__":
    main()
