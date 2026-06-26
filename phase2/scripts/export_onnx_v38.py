"""
v38 モデル ONNX エクスポート

使い方:
  C:\ml\venv\Scripts\python.exe phase2/scripts/export_onnx_v38.py [--model best_wait|best_eae]

出力:
  phase2/models/hand_inference/v38/model.onnx        (best_wait.pt からエクスポート)

修正点 (train_hand_inference_v38.py 内蔵版からの差分):
  - dynamic_axes の n_tokens 軸を 3 → 2 に修正 (disc_tokens shape = B,P,N,TD)
  - Windows cp932 対策: stdout を utf-8 に設定
  - 検証ステップ (PyTorch vs ONNX の誤差チェック) を追加
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# train スクリプトのモデルクラスをインポート
TRAIN_DIR = Path(__file__).parent.parent / "train"
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_hand_inference_v38 import (
    HandInferenceV37,
    CONFIG,
    N_TATSU,
    N_SHANTEN_CLASSES,
)
from enumerate_decompositions import N_BLOCKS_WITH_TATSU as N_BLOCKS

MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v38"


class _WrapForONNX(nn.Module):
    """shanten_logits 以外の5出力に絞ったラッパー"""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, features, disc_tokens, disc_mask):
        logits, _, red_logits, block_logits, wait_logits, _, _, shanten_logits = \
            self.m(features, disc_tokens, disc_mask)
        return logits, red_logits, block_logits, wait_logits, shanten_logits


def load_model(weight_path: Path) -> nn.Module:
    model = HandInferenceV37(
        fixed_dim   = CONFIG["fixed_dim"],
        token_dim   = CONFIG["token_dim"],
        disc_d      = CONFIG["disc_d"],
        disc_nhead  = CONFIG["disc_nhead"],
        disc_layers = CONFIG["disc_layers"],
        d_model     = CONFIG["d_model"],
        nhead       = CONFIG["nhead"],
        num_layers  = CONFIG["num_layers"],
        n_pai       = CONFIG["n_pai"],
        n_count_cls = CONFIG["n_count_cls"],
        n_players   = CONFIG["n_players"],
        n_blocks    = N_BLOCKS,
        n_tatsu     = N_TATSU,
        n_shanten_cls = N_SHANTEN_CLASSES,
        dropout     = 0.0,   # export 時は dropout 無効
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def export(weight_path: Path, out_path: Path, n_tokens_dummy: int = 20):
    print(f"モデルロード: {weight_path}")
    model = load_model(weight_path)
    wrap  = _WrapForONNX(model)
    wrap.eval()

    B, P = 1, 3
    N    = n_tokens_dummy
    F    = CONFIG["fixed_dim"]   # 695
    TD   = CONFIG["token_dim"]   # 44

    dummy_feat = torch.zeros(B, P, F)
    dummy_tok  = torch.zeros(B, P, N, TD)
    # mask: 後半は padding (True)
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
        output_names = ["logits", "red_logits", "block_logits", "wait_logits", "shanten_logits"],
        dynamic_axes = {
            # features: (B, P=3, F=695) — P と F は固定
            "features":       {0: "batch"},
            # disc_tokens: (B, P=3, N, TD=44) — N が可変 (dim=2)
            "disc_tokens":    {0: "batch", 2: "n_tokens"},
            # disc_mask:   (B, P=3, N)         — N が可変 (dim=2)
            "disc_mask":      {0: "batch", 2: "n_tokens"},
            "logits":         {0: "batch"},
            "red_logits":     {0: "batch"},
            "block_logits":   {0: "batch"},
            "wait_logits":    {0: "batch"},
            "shanten_logits": {0: "batch"},
        },
        opset_version = 17,
    )
    print("エクスポート完了")
    return wrap, dummy_feat, dummy_tok, dummy_mask


def merge_external_data(onnx_path: Path):
    """外部データファイル (.onnx.data) を単一 .onnx に統合する（ブラウザ配信用）"""
    ext_path = Path(str(onnx_path) + ".data")
    if not ext_path.exists():
        return  # 外部データなし → 不要

    print(f"外部データを統合中... ({ext_path.name} → {onnx_path.name})")
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=True)
    # save_as_external_data=False で全テンソルをonnx内に埋め込む
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    # 外部データファイルを削除
    ext_path.unlink()
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"統合完了: {onnx_path.name} ({size_mb:.1f} MB, 単一ファイル)")


def validate(wrap, dummy_feat, dummy_tok, dummy_mask, onnx_path: Path):
    """PyTorch 出力 vs ONNX 出力の最大誤差を確認"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[検証スキップ] onnxruntime がインストールされていません")
        return

    print("ONNX 出力検証中...")
    with torch.no_grad():
        pt_outs = wrap(dummy_feat, dummy_tok, dummy_mask)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {
        "features":   dummy_feat.numpy(),
        "disc_tokens": dummy_tok.numpy(),
        "disc_mask":   dummy_mask.numpy(),
    })

    names = ["logits", "red_logits", "block_logits", "wait_logits", "shanten_logits"]
    ok = True
    for name, pt_t, ort_arr in zip(names, pt_outs, ort_outs):
        pt_arr = pt_t.detach().numpy()
        max_err = float(np.abs(pt_arr - ort_arr).max())
        status  = "OK" if max_err < 1e-3 else "NG"
        print(f"  {name}: max_abs_err={max_err:.2e}  [{status}]")
        if max_err >= 1e-3:
            ok = False

    if ok:
        print("検証 PASS: 全出力の最大誤差 < 1e-3")
    else:
        print("警告: 誤差が大きい出力があります。ONNXモデルを確認してください。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["best_wait", "best_eae"], default="best_wait",
                        help="best_wait=model_best_wait.pt (wait_f1最良) / best_eae=model.pt (val_eae最良)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="onnxruntime で出力を検証する")
    args = parser.parse_args()

    if args.model == "best_wait":
        weight_path = MODEL_DIR / "model_best_wait.pt"
    else:
        weight_path = MODEL_DIR / "model.pt"

    out_path = MODEL_DIR / "model.onnx"

    if not weight_path.exists():
        print(f"ERROR: {weight_path} が見つかりません")
        sys.exit(1)

    wrap, df, dt, dm = export(weight_path, out_path)

    # 外部データが生成された場合は単一ファイルに統合（ブラウザ配信用）
    merge_external_data(out_path)

    if args.validate:
        validate(wrap, df, dt, dm, out_path)

    print(f"\n完了: {out_path}")
    print(f"  features:       (1, 3, {CONFIG['fixed_dim']})")
    print(f"  disc_tokens:    (1, 3, N, {CONFIG['token_dim']})  N=可変")
    print(f"  disc_mask:      (1, 3, N)")
    print(f"  logits:         (1, 3, 34, 5)")
    print(f"  red_logits:     (1, 3, 3, 2)")
    print(f"  block_logits:   (1, 3, {N_BLOCKS})")
    print(f"  wait_logits:    (1, 3, {N_TATSU})")
    print(f"  shanten_logits: (1, 3, {N_SHANTEN_CLASSES})")


if __name__ == "__main__":
    main()
