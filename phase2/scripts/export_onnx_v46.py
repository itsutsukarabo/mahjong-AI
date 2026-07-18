"""
v46 モデル ONNX エクスポート（aggression 出力を含む版）

背景:
  train_hand_inference_v46.py 末尾に埋め込まれた ONNX エクスポート（main()内、
  「ONNX エクスポート」セクション）は _Wrap.forward() で
    logits,_,red_logits,block_logits,wait_logits,_,shanten_logits,_ = self.m(...)
  と forward() の5番目の戻り値 aggression_logit を "_" で握り潰しており、
  既存の phase2/models/hand_inference/v46/model.onnx には aggression 出力が
  含まれていない（バイナリ文字列検索で確認済み: logits/red_logits/block_logits/
  wait_logits/shanten_logits の5つのみ）。

  本スクリプトは同じ model.pt（state_dict, best val_eae = ep185相当）から、
  aggression_logit を追加した6出力で別名ファイルとして再エクスポートする。
  既存の model.onnx は上書きしない。

使い方:
  C:\\ml\\venv\\Scripts\\python.exe phase2/scripts/export_onnx_v46.py

export_onnx_v38.py からの差分:
  - インポート元: train_hand_inference_v38 → train_hand_inference_v46
  - fixed_dim: 695ハードコード相当 → CONFIG["fixed_dim"] (674, v46で変更)
  - token_dim: 44ハードコード → CONFIG["token_dim"] (45, v45でdanger_level追加分)
  - output_names に aggression_logit を追加（forward()の5番目の戻り値、
    形状 (B,) — 視点プレイヤー1名分のスカラー。aggression_head内で既にTanh適用済みのため
    ONNX側で追加のsigmoid/tanhは不要）
  - --model 引数（best_wait/best_eae選択）を廃止。v46は model.pt 一種類のみ
    （model_best_wait.pt に相当するファイルは存在しない）
  - 実データ1サンプルでの追加検証（validate_real_sample）を新設
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# train スクリプトのモデルクラス・CONFIGをインポート
TRAIN_DIR = Path(__file__).parent.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_hand_inference_v46 import (
    HandInferenceV37,
    CONFIG,
    N_TATSU,
    N_SHANTEN_CLASSES,
)

MODEL_DIR   = Path(__file__).parent.parent / "models" / "hand_inference" / "v46"
WEIGHT_PATH = MODEL_DIR / "model.pt"  # 明示: best val_eae (ep185相当) の state_dict。checkpoint.pt(最新epoch, optimizer付き)は使わない
OUT_PATH    = MODEL_DIR / "model_aggression_candidate.onnx"  # 既存 model.onnx は上書きしない
DATA_PATH   = Path(__file__).parent.parent / "data" / "features" / "hand_inference_v46.ndjson"


def check_config_drift():
    """config.json とインポートしたCONFIGの不一致を検知する（既知の次元バグ再発防止）"""
    cfg_json_path = MODEL_DIR / "config.json"
    if not cfg_json_path.exists():
        return
    disk_cfg = json.loads(cfg_json_path.read_text())
    for k in ("fixed_dim", "token_dim", "d_model", "n_blocks", "disc_d"):
        if k in disk_cfg and k in CONFIG and disk_cfg[k] != CONFIG[k]:
            print(f"[警告] config.json の {k}={disk_cfg[k]} と "
                  f"train_hand_inference_v46.CONFIG の {k}={CONFIG[k]} が不一致")


class _WrapForONNX(nn.Module):
    """aggression_logit を含む6出力に絞ったラッパー"""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, features, disc_tokens, disc_mask):
        logits, _logits_raw, red_logits, block_logits, wait_logits, \
            aggression_logit, shanten_logits, _yaku_logits = \
            self.m(features, disc_tokens, disc_mask)
        return logits, red_logits, block_logits, wait_logits, aggression_logit, shanten_logits


def load_model(weight_path: Path) -> nn.Module:
    if not weight_path.exists():
        print(f"ERROR: {weight_path} が見つかりません")
        sys.exit(1)
    model = HandInferenceV37(
        fixed_dim     = CONFIG["fixed_dim"],
        token_dim     = CONFIG["token_dim"],
        disc_d        = CONFIG["disc_d"],
        disc_nhead    = CONFIG["disc_nhead"],
        disc_layers   = CONFIG["disc_layers"],
        d_model       = CONFIG["d_model"],
        nhead         = CONFIG["nhead"],
        num_layers    = CONFIG["num_layers"],
        n_pai         = CONFIG["n_pai"],
        n_count_cls   = CONFIG["n_count_cls"],
        n_players     = CONFIG["n_players"],
        n_blocks      = CONFIG["n_blocks"],
        n_tatsu       = N_TATSU,
        n_shanten_cls = N_SHANTEN_CLASSES,
        dropout       = 0.0,  # export時はdropout無効
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def export(model: nn.Module, out_path: Path, n_tokens_dummy: int = 20):
    wrap = _WrapForONNX(model)
    wrap.eval()

    B, P = 1, 3
    N  = n_tokens_dummy
    F  = CONFIG["fixed_dim"]
    TD = CONFIG["token_dim"]

    dummy_feat = torch.zeros(B, P, F)
    dummy_tok  = torch.zeros(B, P, N, TD)
    dummy_mask = torch.cat([
        torch.zeros(B, P, N // 2, dtype=torch.bool),
        torch.ones( B, P, N - N // 2, dtype=torch.bool),
    ], dim=2)

    print(f"ONNX エクスポート中... -> {out_path}")
    torch.onnx.export(
        wrap,
        (dummy_feat, dummy_tok, dummy_mask),
        str(out_path),
        input_names  = ["features", "disc_tokens", "disc_mask"],
        output_names = ["logits", "red_logits", "block_logits", "wait_logits",
                         "aggression_logit", "shanten_logits"],
        dynamic_axes = {
            "features":         {0: "batch"},
            "disc_tokens":      {0: "batch", 2: "n_tokens"},
            "disc_mask":        {0: "batch", 2: "n_tokens"},
            "logits":           {0: "batch"},
            "red_logits":       {0: "batch"},
            "block_logits":     {0: "batch"},
            "wait_logits":      {0: "batch"},
            "aggression_logit": {0: "batch"},  # (B,) 視点プレイヤー1名分のスカラー
            "shanten_logits":   {0: "batch"},
        },
        opset_version = 17,
    )
    print("エクスポート完了")
    return wrap, dummy_feat, dummy_tok, dummy_mask


def merge_external_data(onnx_path: Path):
    """外部データファイル (.onnx.data) を単一 .onnx に統合する（ブラウザ配信用）"""
    ext_path = Path(str(onnx_path) + ".data")
    if not ext_path.exists():
        print("外部データなし（単一ファイルとして出力されました）")
        return
    try:
        import onnx
    except ImportError:
        print(f"[警告] onnx パッケージ未インストールのため外部データ統合をスキップ。"
              f" {onnx_path.name} と {ext_path.name} を対で保管してください。")
        return

    print(f"外部データを統合中... ({ext_path.name} → {onnx_path.name})")
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    ext_path.unlink()
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"統合完了: {onnx_path.name} ({size_mb:.1f} MB, 単一ファイル)")


def validate_dummy(wrap, dummy_feat, dummy_tok, dummy_mask, onnx_path: Path):
    """ダミー入力: PyTorch出力 vs ONNX出力の最大誤差を確認"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[検証スキップ] onnxruntime がインストールされていません")
        return None

    print("\n=== ダミー入力による検証 ===")
    with torch.no_grad():
        pt_outs = wrap(dummy_feat, dummy_tok, dummy_mask)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {
        "features":    dummy_feat.numpy(),
        "disc_tokens": dummy_tok.numpy(),
        "disc_mask":   dummy_mask.numpy(),
    })

    names = ["logits", "red_logits", "block_logits", "wait_logits", "aggression_logit", "shanten_logits"]
    ok = True
    for name, pt_t, ort_arr in zip(names, pt_outs, ort_outs):
        pt_arr = pt_t.detach().numpy()
        max_err = float(np.abs(pt_arr - ort_arr).max())
        status  = "OK" if max_err < 1e-3 else "NG"
        print(f"  {name}: shape={tuple(ort_arr.shape)}  max_abs_err={max_err:.2e}  [{status}]")
        if max_err >= 1e-3:
            ok = False

    agg_val  = float(ort_outs[names.index("aggression_logit")].reshape(-1)[0])
    in_range = -1.0 <= agg_val <= 1.0
    print(f"  aggression_logit (dummy zero-input) = {agg_val:.4f}  "
          f"range_check(-1..1)={'OK' if in_range else 'NG'}")

    if ok and in_range:
        print("検証 PASS: 全出力の最大誤差 < 1e-3、aggressionは範囲内")
    else:
        print("警告: 誤差または範囲に問題があります。")
    return ok and in_range


def validate_real_sample(model: nn.Module, onnx_path: Path, data_path: Path):
    """実データ1サンプルで PyTorch と ONNX の aggression 出力を比較"""
    if not data_path.exists():
        print(f"[実データ検証スキップ] {data_path} が見つかりません")
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        print("[実データ検証スキップ] onnxruntime がインストールされていません")
        return None

    print("\n=== 実データ1サンプルによる検証 ===")
    F  = CONFIG["fixed_dim"]
    TD = CONFIG["token_dim"]

    with open(data_path, encoding="utf-8") as f:
        line = f.readline()
        while line and not line.strip():
            line = f.readline()
    if not line:
        print("[実データ検証スキップ] ndjsonが空です")
        return None
    sample = json.loads(line)

    feat = np.array(sample["features"], dtype=np.float32).reshape(1, 3, F)
    raw_tok = sample.get("discard_tokens", [[], [], []])
    max_len = max(1, max(len(raw_tok[p]) for p in range(3)))
    tok  = np.zeros((1, 3, max_len, TD), dtype=np.float32)
    mask = np.ones((1, 3, max_len), dtype=bool)
    for p in range(3):
        n = len(raw_tok[p])
        if n > 0:
            tok[0, p, :n]  = np.array(raw_tok[p], dtype=np.float32)
            mask[0, p, :n] = False

    feat_t = torch.from_numpy(feat)
    tok_t  = torch.from_numpy(tok)
    mask_t = torch.from_numpy(mask)

    with torch.no_grad():
        out = model(feat_t, tok_t, mask_t)
    pt_agg = float(out[5].reshape(-1)[0])  # forward()の5番目 = aggression_logit

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {
        "features": feat, "disc_tokens": tok, "disc_mask": mask,
    })
    names = ["logits", "red_logits", "block_logits", "wait_logits", "aggression_logit", "shanten_logits"]
    onnx_agg = float(ort_outs[names.index("aggression_logit")].reshape(-1)[0])

    label    = float(sample.get("label_aggression", 0.0))
    diff     = abs(pt_agg - onnx_agg)
    in_range = -1.0 <= onnx_agg <= 1.0
    print(f"  PyTorch aggression = {pt_agg:.4f}")
    print(f"  ONNX    aggression = {onnx_agg:.4f}")
    print(f"  |diff|             = {diff:.2e}  [{'OK' if diff < 1e-3 else 'NG'}]")
    print(f"  range_check(-1..1) = {'OK' if in_range else 'NG'}")
    print(f"  参考: このサンプルの label_aggression = {label:.4f}")
    return diff < 1e-3 and in_range


def main():
    check_config_drift()

    print(f"モデルロード: {WEIGHT_PATH}")
    model = load_model(WEIGHT_PATH)

    wrap, df, dt, dm = export(model, OUT_PATH)
    merge_external_data(OUT_PATH)

    ok_dummy = validate_dummy(wrap, df, dt, dm, OUT_PATH)
    ok_real  = validate_real_sample(model, OUT_PATH, DATA_PATH)

    print(f"\n完了: {OUT_PATH}")
    print(f"  features:          (1, 3, {CONFIG['fixed_dim']})")
    print(f"  disc_tokens:       (1, 3, N, {CONFIG['token_dim']})  N=可変")
    print(f"  disc_mask:         (1, 3, N)")
    print(f"  logits:            (1, 3, 34, 5)")
    print(f"  red_logits:        (1, 3, 3, 2)")
    print(f"  block_logits:      (1, 3, {CONFIG['n_blocks']})")
    print(f"  wait_logits:       (1, 3, {N_TATSU})")
    print(f"  aggression_logit:  (1,)          ← 新規追加")
    print(f"  shanten_logits:    (1, 3, {N_SHANTEN_CLASSES})")

    if ok_dummy is False or ok_real is False:
        print("\n[結論] 検証NG。既存 model.onnx を差し替えないこと。要確認・再エクスポート。")
        sys.exit(1)
    else:
        print("\n[結論] 検証OK。aggression出力を含むONNXの再エクスポートに成功。")
        print(f"       既存の {MODEL_DIR / 'model.onnx'} は変更していません。")
        print(f"       配置を決めるまで {OUT_PATH.name} のまま保持してください。")


if __name__ == "__main__":
    main()
