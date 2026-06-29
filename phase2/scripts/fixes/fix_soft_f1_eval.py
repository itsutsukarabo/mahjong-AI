"""
fix_soft_f1_eval: wait評価指標を threshold ベースから soft F1 に変更 (v39 → v40)

変更点:
  1. eval_wait_metrics に soft_f1 を追加
       soft_tp = 正解タイルへの確率合計
       soft_fp = 不正解タイルへの確率合計
       soft_fn = len(true_waits) - soft_tp
       soft_f1 = 2*soft_tp / (2*soft_tp + soft_fp + soft_fn)
  2. model_best_wait.pt の保存基準を wait_f1 → soft_f1 に変更
  3. コンソール出力・train_log に soft_f1 を追加（wait_f1 は参考値として残す）

根拠:
  threshold=0.3 でバイナリ化する wait_f1 は、不確実な状況で正直に
  確率を分散させた出力に低スコアを与えるバイアスがある。
  例: 69m待ち50%・残り50%分散という正直な出力が threshold 未満になると F1=0。
  soft_f1 は確率値をそのまま重みとして使うため、このバイアスがない。
"""

# ---- eval_wait_metrics 関数の置換 ----

_OLD_EVAL_FUNC = '''\
@torch.no_grad()
def eval_wait_metrics(model, loader, device, threshold=WAIT_EVAL_THRESH):
    model.eval()
    prec_list=[]; recall_list=[]; f1_list=[]; hit_list=[]; top1_list=[]
    for batch in loader:
        features = batch[0].to(device)
        disc_tok  = batch[1].to(device)
        disc_mask = batch[2].to(device)
        labels = batch[3]; labels_shanten = batch[6]
        out = model(features, disc_tok, disc_mask)
        wait_tatsu_probs = torch.sigmoid(out[4]).cpu().numpy()
        hand_np=labels.numpy(); sh_np=labels_shanten.numpy()
        for b in range(len(features)):
            for p in range(3):
                if sh_np[b,p] != 0: continue
                counts34=hand_np[b,p].tolist(); total=int(sum(counts34))
                if total==0 or total%3!=1: continue
                true_waits=set(compute_true_waits(counts34,(13-total)//3))
                if not true_waits: continue
                tile_probs=tatsu_probs_to_tile_probs(wait_tatsu_probs[b,p])
                pred_waits={t for t in range(N_PAI) if tile_probs[t]>=threshold}
                top1=int(np.argmax(tile_probs))
                tp=len(pred_waits&true_waits); fp=len(pred_waits-true_waits); fn=len(true_waits-pred_waits)
                prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
                recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1=2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0
                prec_list.append(prec); recall_list.append(recall); f1_list.append(f1)
                hit_list.append(1.0 if tp>0 else 0.0)
                top1_list.append(1.0 if top1 in true_waits else 0.0)
    if not f1_list:
        return {"wait_f1":0.0,"wait_prec":0.0,"wait_recall":0.0,"wait_hit_rate":0.0,"wait_top1_acc":0.0,"n_tenpai":0}
    return {
        "wait_f1":       round(float(np.mean(f1_list)),4),
        "wait_prec":     round(float(np.mean(prec_list)),4),
        "wait_recall":   round(float(np.mean(recall_list)),4),
        "wait_hit_rate": round(float(np.mean(hit_list)),4),
        "wait_top1_acc": round(float(np.mean(top1_list)),4),
        "n_tenpai":      len(f1_list),
    }'''

_NEW_EVAL_FUNC = '''\
@torch.no_grad()
def eval_wait_metrics(model, loader, device, threshold=WAIT_EVAL_THRESH):
    model.eval()
    soft_f1_list=[]; f1_list=[]; hit_list=[]; top1_list=[]
    for batch in loader:
        features = batch[0].to(device)
        disc_tok  = batch[1].to(device)
        disc_mask = batch[2].to(device)
        labels = batch[3]; labels_shanten = batch[6]
        out = model(features, disc_tok, disc_mask)
        wait_tatsu_probs = torch.sigmoid(out[4]).cpu().numpy()
        hand_np=labels.numpy(); sh_np=labels_shanten.numpy()
        for b in range(len(features)):
            for p in range(3):
                if sh_np[b,p] != 0: continue
                counts34=hand_np[b,p].tolist(); total=int(sum(counts34))
                if total==0 or total%3!=1: continue
                true_waits=set(compute_true_waits(counts34,(13-total)//3))
                if not true_waits: continue
                tile_probs=tatsu_probs_to_tile_probs(wait_tatsu_probs[b,p])
                top1=int(np.argmax(tile_probs))

                # ---- soft F1（閾値なし・確率値を重みとして使用） ----
                soft_tp = sum(tile_probs[t] for t in true_waits)
                soft_fp = sum(tile_probs[t] for t in range(N_PAI) if t not in true_waits)
                soft_fn = len(true_waits) - soft_tp
                denom   = 2*soft_tp + soft_fp + soft_fn
                soft_f1 = (2*soft_tp / denom) if denom > 0 else 0.0
                soft_f1_list.append(soft_f1)

                # ---- 参考: threshold ベース wait_f1 ----
                pred_waits={t for t in range(N_PAI) if tile_probs[t]>=threshold}
                tp=len(pred_waits&true_waits); fp=len(pred_waits-true_waits); fn=len(true_waits-pred_waits)
                prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
                recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1=2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0
                f1_list.append(f1)

                hit_list.append(1.0 if tp>0 else 0.0)
                top1_list.append(1.0 if top1 in true_waits else 0.0)
    if not soft_f1_list:
        return {"soft_f1":0.0,"wait_f1":0.0,"wait_hit_rate":0.0,"wait_top1_acc":0.0,"n_tenpai":0}
    return {
        "soft_f1":       round(float(np.mean(soft_f1_list)),4),
        "wait_f1":       round(float(np.mean(f1_list)),4),
        "wait_hit_rate": round(float(np.mean(hit_list)),4),
        "wait_top1_acc": round(float(np.mean(top1_list)),4),
        "n_tenpai":      len(soft_f1_list),
    }'''


# ---- コンソール出力・保存基準の置換 ----

_OLD_PRINT = (
    '        print(f"epoch {epoch:3d}  nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}"\n'
    '              f"  wait_f1={wait_m[\'wait_f1\']:.4f}  hit={wait_m[\'wait_hit_rate\']:.4f}"\n'
    '              f"  top1={wait_m[\'wait_top1_acc\']:.4f}", flush=True)'
)

_NEW_PRINT = (
    '        print(f"epoch {epoch:3d}  nll={train_nll:.4f}  val_eae={val_eae:.4f}  val_acc={val_acc:.4f}"\n'
    '              f"  soft_f1={wait_m[\'soft_f1\']:.4f}  wait_f1={wait_m[\'wait_f1\']:.4f}"\n'
    '              f"  hit={wait_m[\'wait_hit_rate\']:.4f}  top1={wait_m[\'wait_top1_acc\']:.4f}", flush=True)'
)

_OLD_SAVE = '''\
        if wait_m["wait_f1"] > best_wait_f1:
            best_wait_f1 = wait_m["wait_f1"]
            torch.save(model.state_dict(), MODEL_DIR/"model_best_wait.pt")
            print(f"  [saved best_wait f1={best_wait_f1:.4f}]", flush=True)'''

_NEW_SAVE = '''\
        if wait_m["soft_f1"] > best_wait_f1:
            best_wait_f1 = wait_m["soft_f1"]
            torch.save(model.state_dict(), MODEL_DIR/"model_best_wait.pt")
            print(f"  [saved best_soft_f1={best_wait_f1:.4f}]", flush=True)'''


def patch_train_script(code: str, base_ver: int, target_ver: int) -> str:
    # 1. eval_wait_metrics 関数を置換
    if 'soft_f1_list' not in code:
        assert _OLD_EVAL_FUNC in code, 'eval_wait_metrics の検索文字列が見つかりません'
        code = code.replace(_OLD_EVAL_FUNC, _NEW_EVAL_FUNC, 1)

    # 2. コンソール出力に soft_f1 を追加
    if "soft_f1={wait_m['soft_f1']" not in code:
        assert _OLD_PRINT in code, 'print 行の検索文字列が見つかりません'
        code = code.replace(_OLD_PRINT, _NEW_PRINT, 1)

    # 3. 保存基準を soft_f1 に変更
    if 'wait_m["soft_f1"] > best_wait_f1' not in code:
        assert _OLD_SAVE in code, '保存基準の検索文字列が見つかりません'
        code = code.replace(_OLD_SAVE, _NEW_SAVE, 1)

    return code
