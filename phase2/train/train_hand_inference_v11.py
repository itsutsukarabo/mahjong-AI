"""
手牌類推モデル v11: v10 にブロックヘッド（刻子/順子/対子）を追加

入力 (374次元): v10 と同一

アーキテクチャ:
  v10 バックボーン + block heads:
    triplet_head: Linear(d_model, 1) → sigmoid → P(刻子|聴牌) [34次元]
    seq_head:     Linear(d_model, 1) → sigmoid → P(順子|聴牌) [21次元; m1-7/p1-7/s1-7]
    pair_head:    Linear(d_model, 1) → sigmoid → P(対子|聴牌) [34次元]

損失関数:
  loss = CE + λ_gs*MSE(sum) + λ_red_ce*CE_red + λ_red_cons*relu_cons
       + λ_block * (BCE_triplet + BCE_seq + BCE_pair) / 3
  block 損失は tenpai サンプルのみで計算 (tenpai_mask)

label_block フォーマット (89次元 flat):
  [0:34]  triplet ソフトラベル
  [34:55] seq ソフトラベル (21次元)
  [55:89] pair ソフトラベル (34次元)

ONNX 出力: logits, red_logits, triplet_logits, seq_logits, pair_logits

評価 (TICKET-034): remaining バケット別 accuracy を eval_result.json に追記
"""

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v11"

CONFIG = {
    "input_dim":    374,
    "d_model":      256,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "dropout":      0.1,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,
    "epochs":       200,
    "early_stop_patience": 7,
    "lambda_block": 0.3,
}

VISIBLE_OFFSET     = 185
REMAINING_OFFSET   = 94
GPU_TEMP_THRESHOLD = 65
GPU_COOL_INTERVAL  = 20

LAMBDA_GLOBAL_SUM  = 0.05
LAMBDA_RED_CE      = 0.3
LAMBDA_RED_CONS    = 0.1

BLOCK_DIM = 89  # triplet(34) + seq(21) + pair(34)

REMAINING_BUCKETS = {
    "early":   (51, 9999),
    "mid":     (31,   50),
    "late":    (11,   30),
    "endgame": ( 0,   10),
}

# 順子トークンのタイルインデックス (m1-7: 0-6, p1-7: 9-15, s1-7: 18-24)
SEQ_TILE_INDICES = list(range(0, 7)) + list(range(9, 16)) + list(range(18, 25))


def gpu_temp():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def wait_for_cool():
    temp = gpu_temp()
    if temp <= GPU_TEMP_THRESHOLD:
        return
    print(f"  [thermal] GPU {temp}C > {GPU_TEMP_THRESHOLD}C - cooling...", flush=True)
    while True:
        time.sleep(GPU_COOL_INTERVAL)
        temp = gpu_temp()
        print(f"  [thermal] GPU {temp}C", flush=True)
        if temp <= GPU_TEMP_THRESHOLD:
            print("  [thermal] cool enough, resuming", flush=True)
            break


# ---- データセット ----

def make_block_tensor(label_block):
    """label_block dict → 89次元 float tensor; None の場合はゼロベクトルを返す"""
    if label_block is None:
        return torch.zeros(BLOCK_DIM, dtype=torch.float32), False
    t = (label_block["triplet"] + label_block["seq"] + label_block["pair"])
    return torch.tensor(t, dtype=torch.float32), True


class HandInferenceDataset(Dataset):
    def __init__(self, data):
        self.features   = torch.tensor(
            [s["features"] for s in data], dtype=torch.float32
        )
        self.labels     = torch.tensor(
            [s["label_hand"] for s in data], dtype=torch.long
        ).clamp(0, 4)
        self.labels_red = torch.tensor(
            [s["label_red"] for s in data], dtype=torch.long
        )

        block_tensors = []
        tenpai_masks  = []
        for s in data:
            bt, mask = make_block_tensor(s.get("label_block"))
            block_tensors.append(bt)
            tenpai_masks.append(mask)
        self.labels_block  = torch.stack(block_tensors)           # (N, 89)
        self.tenpai_mask   = torch.tensor(tenpai_masks, dtype=torch.bool)  # (N,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.labels_red[idx],
            self.labels_block[idx],
            self.tenpai_mask[idx],
        )


# ---- モデル ----

class HandInferenceV11(nn.Module):

    SEQ_TILE_IDX = torch.tensor(SEQ_TILE_INDICES)

    def __init__(self, input_dim, d_model, nhead, num_layers, n_pai, n_count_cls, dropout):
        super().__init__()
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls

        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        self.tile_embed   = nn.Embedding(n_pai, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head         = nn.Linear(d_model, n_count_cls)
        self.red_head     = nn.Linear(d_model, 2)

        # block heads
        self.triplet_head = nn.Linear(d_model, 1)
        self.pair_head    = nn.Linear(d_model, 1)
        self.seq_head     = nn.Linear(d_model, 1)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))  # 5m/5p/5s
        self.register_buffer("seq_tile_idx", self.SEQ_TILE_IDX)

    def forward(self, x):
        g = self.global_encoder(x)                                            # [B, d_model]

        tile_emb = self.tile_embed(self.tile_ids)                             # [34, d_model]
        vis = x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai].unsqueeze(-1)  # [B, 34, 1]
        vis_emb = self.visible_proj(vis)                                      # [B, 34, d_model]

        tokens = g.unsqueeze(1) + tile_emb.unsqueeze(0) + vis_emb             # [B, 34, d_model]
        out    = self.transformer(tokens)                                      # [B, 34, d_model]
        logits_raw = self.head(out)                                            # [B, 34, 5]

        vis_raw = (x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai] * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)
        mask = self.count_range > hidden_remaining.unsqueeze(-1)
        logits = logits_raw.masked_fill(mask, float('-inf'))

        red_tokens = out[:, self.red_tile_idx, :]                             # [B, 3, d_model]
        red_logits = self.red_head(red_tokens)                                # [B, 3, 2]

        # block heads
        triplet_logits = self.triplet_head(out).squeeze(-1)                   # [B, 34]
        pair_logits    = self.pair_head(out).squeeze(-1)                      # [B, 34]
        seq_tokens     = out[:, self.seq_tile_idx, :]                         # [B, 21, d_model]
        seq_logits     = self.seq_head(seq_tokens).squeeze(-1)                # [B, 21]

        return logits, logits_raw, red_logits, triplet_logits, seq_logits, pair_logits


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_ce     = 0.0
    total_global = 0.0
    count_vals   = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx     = model.red_tile_idx

    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features    = features.to(device)
        labels      = labels.to(device)
        labels_red  = labels_red.to(device)
        labels_block = labels_block.to(device)
        tenpai_mask  = tenpai_mask.to(device)
        optimizer.zero_grad()

        _, logits_raw, red_logits, tri_logits, seq_logits, pair_logits = model(features)

        loss_ce = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))

        probs          = F.softmax(logits_raw, dim=-1)
        expected_total = (probs * count_vals).sum(-1).sum(-1)
        target_total   = labels.float().sum(-1)
        loss_global    = F.mse_loss(expected_total, target_total)

        loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

        prob_has_red  = F.softmax(red_logits, dim=-1)[:, :, 1]
        prob_cnt_ge1  = 1 - F.softmax(logits_raw[:, five_idx, :], dim=-1)[:, :, 0]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        # block 損失（tenpai サンプルのみ）
        block_loss = 0.0
        n_tenpai = tenpai_mask.sum().item()
        if n_tenpai > 0:
            tri = tri_logits[tenpai_mask]                   # (N, 34)
            seq = seq_logits[tenpai_mask]                   # (N, 21)
            pai = pair_logits[tenpai_mask]                  # (N, 34)
            t_tri = labels_block[tenpai_mask, :34]
            t_seq = labels_block[tenpai_mask, 34:55]
            t_pai = labels_block[tenpai_mask, 55:89]
            block_loss = (F.binary_cross_entropy_with_logits(tri, t_tri) +
                          F.binary_cross_entropy_with_logits(seq, t_seq) +
                          F.binary_cross_entropy_with_logits(pai, t_pai)) / 3

        loss = (loss_ce
                + LAMBDA_GLOBAL_SUM * loss_global
                + LAMBDA_RED_CE     * loss_red_ce
                + LAMBDA_RED_CONS   * loss_red_cons
                + CONFIG["lambda_block"] * block_loss)
        loss.backward()
        optimizer.step()

        total_ce     += loss_ce.item()     * len(features)
        total_global += loss_global.item() * len(features)

    n = len(loader.dataset)
    return total_ce / n, total_global / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features, labels = features.to(device), labels.to(device)
        logits, logits_raw, _, _, _, _ = model(features)
        loss = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))
        total_loss += loss.item() * len(features)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def eval_by_remaining(model, loader, device):
    model.eval()
    buckets = {k: {"correct": 0, "total": 0, "loss_sum": 0.0} for k in REMAINING_BUCKETS}

    for features, labels, labels_red, labels_block, tenpai_mask in loader:
        features, labels = features.to(device), labels.to(device)
        logits, logits_raw, _, _, _, _ = model(features)

        preds = logits.argmax(dim=-1)
        correct_mask = (preds == labels)
        loss_per_sample = F.cross_entropy(
            logits_raw.reshape(-1, model.n_count_cls),
            labels.reshape(-1),
            reduction="none",
        ).reshape(len(features), -1).mean(dim=-1)

        remaining_vals = (features[:, REMAINING_OFFSET] * 70).round().long().cpu()

        for i, rem in enumerate(remaining_vals):
            rem = rem.item()
            for name, (lo, hi) in REMAINING_BUCKETS.items():
                if lo <= rem <= hi:
                    b = buckets[name]
                    b["correct"] += correct_mask[i].sum().item()
                    b["total"]   += correct_mask[i].numel()
                    b["loss_sum"] += loss_per_sample[i].item()
                    break

    result = {}
    for name, b in buckets.items():
        n = b["total"] // model.n_pai if model.n_pai > 0 else 1
        result[name] = {
            "acc":  b["correct"] / b["total"] if b["total"] > 0 else 0.0,
            "loss": b["loss_sum"] / (n or 1),
            "n":    n,
        }
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ndjson_path = DATA_DIR / "hand_inference.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        sys.exit(1)
    print(f"読み込み中: {ndjson_path}")
    with open(ndjson_path, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    print(f"総サンプル数: {len(all_data)}")

    sample_dim = len(all_data[0]["features"])
    if sample_dim != CONFIG["input_dim"]:
        print(f"次元数不一致: expected {CONFIG['input_dim']}, got {sample_dim}")
        print("extract_features.js → add_yaku_features.py → add_tenpai_features.py の順に実行してください")
        sys.exit(1)

    if "label_block" not in all_data[0]:
        print("label_block フィールドがありません。extract_features.js を再実行してください")
        sys.exit(1)

    n_tenpai = sum(1 for s in all_data if s.get("label_block") is not None)
    print(f"tenpai サンプル数 (label_block あり): {n_tenpai} ({n_tenpai/len(all_data)*100:.1f}%)")

    random.seed(42)
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_data = all_data[:n_train]
    val_data   = all_data[n_train:n_train + n_val]
    test_data  = all_data[n_train + n_val:]
    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(HandInferenceDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(HandInferenceDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(HandInferenceDataset(test_data),  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV11(
        input_dim   = CONFIG["input_dim"],
        d_model     = CONFIG["d_model"],
        nhead       = CONFIG["nhead"],
        num_layers  = CONFIG["num_layers"],
        n_pai       = CONFIG["n_pai"],
        n_count_cls = CONFIG["n_count_cls"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = math.inf
    patience_cnt  = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_ce, train_gs = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc  = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d}  train_loss={train_ce:.4f}  gs={train_gs:.3f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)
        log_entry = {"epoch": epoch, "train_loss": train_ce, "gs_loss": train_gs,
                     "val_loss": val_loss, "val_acc": val_acc}
        with open(MODEL_DIR / "train_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        wait_for_cool()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["early_stop_patience"]:
                print("early stopping")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    bucket_result = eval_by_remaining(model, test_loader, device)
    print("バケット別精度:")
    for name, b in bucket_result.items():
        print(f"  {name:8s}: acc={b['acc']:.4f}  loss={b['loss']:.4f}  n={b['n']}")

    eval_result = {
        "test_loss": test_loss,
        "test_acc":  test_acc,
        "by_remaining_bucket": bucket_result,
        "config": CONFIG,
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval().cpu()
    dummy = torch.zeros(1, CONFIG["input_dim"])

    class _OnnxWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits, tri_logits, seq_logits, pair_logits = self.m(x)
            return logits, red_logits, tri_logits, seq_logits, pair_logits

    torch.onnx.export(
        _OnnxWrapper(model), dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "red_logits", "triplet_logits", "seq_logits", "pair_logits"],
        dynamic_axes={
            "features":        {0: "batch_size"},
            "logits":          {0: "batch_size"},
            "red_logits":      {0: "batch_size"},
            "triplet_logits":  {0: "batch_size"},
            "seq_logits":      {0: "batch_size"},
            "pair_logits":     {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
