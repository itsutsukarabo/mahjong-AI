"""
手牌類推モデル v3: Per-tile アーキテクチャ + Ordinal Soft Labels

入力 (777次元):
  Per-tile (34タイル × 22次元):
    tile_identity(14) + self_count(1) + target_discard(1) + visible(1) + neighbor_vis(4) + suit_disc(1)
  グローバル (29次元):
    score(11) + game(9) + riichi(1) + target_meld_type(4) + self_meld_type(4)

出力:
  他家の手牌確率 (34牌 × 5クラス: 0/1/2/3/4枚)
"""

import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v3"

CONFIG = {
    "n_pai":          34,
    "per_tile_dim":   22,
    "global_dim":     29,
    "input_dim":      34 * 22 + 29,   # 777
    "shared_hidden":  [128, 64],
    "n_count_cls":    5,
    "dropout":        0.2,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "batch_size":     512,
    "epochs":         60,
    "early_stop_patience": 7,
    "ordinal_sigma":  0.7,
}


# ---- Ordinal Soft Labels ----

def make_soft_label_matrix(n_classes, sigma):
    """正解クラスから離れるほど確率が下がる soft label 行列を生成"""
    mat = torch.zeros(n_classes, n_classes)
    for c in range(n_classes):
        for j in range(n_classes):
            d = abs(c - j)
            mat[c, j] = math.exp(-d * d / (2 * sigma * sigma))
        mat[c] /= mat[c].sum()
    return mat


# ---- データセット ----

class HandInferenceDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            [s["features"] for s in data], dtype=torch.float32
        )
        self.labels = torch.tensor(
            [s["label_hand"] for s in data], dtype=torch.long
        ).clamp(0, 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---- モデル (Per-tile Shared MLP) ----

class PerTileHandInferenceModel(nn.Module):
    def __init__(self, per_tile_dim, global_dim, shared_hidden, n_pai, n_count_cls, dropout):
        super().__init__()
        tile_feat_dim = per_tile_dim + global_dim  # 22 + 29 = 51
        layers = []
        prev = tile_feat_dim
        for h in shared_hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.shared_mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_count_cls)
        self.n_pai       = n_pai
        self.per_tile_dim = per_tile_dim
        self.global_dim  = global_dim
        self.n_count_cls = n_count_cls

    def forward(self, x):
        # x: (B, 34*per_tile_dim + global_dim) = (B, 777)
        B = x.shape[0]
        per_tile_part = x[:, : self.n_pai * self.per_tile_dim]          # (B, 748)
        global_part   = x[:, self.n_pai * self.per_tile_dim :]          # (B, 29)

        per_tile  = per_tile_part.reshape(B, self.n_pai, self.per_tile_dim)  # (B, 34, 22)
        global_ex = global_part.unsqueeze(1).expand(-1, self.n_pai, -1)      # (B, 34, 29)

        x_tile = torch.cat([per_tile, global_ex], dim=-1)               # (B, 34, 51)
        x_flat = x_tile.reshape(-1, self.per_tile_dim + self.global_dim) # (B*34, 51)

        h   = self.shared_mlp(x_flat)                                    # (B*34, last_h)
        out = self.head(h)                                                # (B*34, 5)
        return out.reshape(B, self.n_pai, self.n_count_cls)              # (B, 34, 5)


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, soft_matrix, device):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)                             # (B, 34, 5)
        log_probs = F.log_softmax(logits, dim=-1)           # (B, 34, 5)
        soft_tgt  = soft_matrix[labels.view(-1)]            # (B*34, 5)
        log_flat  = log_probs.view(-1, model.n_count_cls)   # (B*34, 5)
        loss = F.kl_div(log_flat, soft_tgt, reduction='batchmean')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, soft_matrix, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        log_probs = F.log_softmax(logits, dim=-1)
        soft_tgt  = soft_matrix[labels.view(-1)]
        log_flat  = log_probs.view(-1, model.n_count_cls)
        loss = F.kl_div(log_flat, soft_tgt, reduction='batchmean')
        total_loss += loss.item() * len(features)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
    acc = correct / total
    return total_loss / len(loader.dataset), acc


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

    # 次元数チェック
    sample_dim = len(all_data[0]["features"])
    if sample_dim != CONFIG["input_dim"]:
        print(f"次元数不一致: expected {CONFIG['input_dim']}, got {sample_dim}")
        print("extract_features.js を再実行してください")
        sys.exit(1)

    random.seed(42)
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_data = all_data[:n_train]
    val_data   = all_data[n_train:n_train + n_val]
    test_data  = all_data[n_train + n_val:]
    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    train_loader = DataLoader(HandInferenceDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(HandInferenceDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(HandInferenceDataset(test_data),  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    soft_matrix = make_soft_label_matrix(CONFIG["n_count_cls"], CONFIG["ordinal_sigma"]).to(device)
    print("Ordinal soft label matrix:")
    for i, row in enumerate(soft_matrix):
        print(f"  count={i}: {[f'{v:.3f}' for v in row.tolist()]}")

    model = PerTileHandInferenceModel(
        per_tile_dim  = CONFIG["per_tile_dim"],
        global_dim    = CONFIG["global_dim"],
        shared_hidden = CONFIG["shared_hidden"],
        n_pai         = CONFIG["n_pai"],
        n_count_cls   = CONFIG["n_count_cls"],
        dropout       = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = math.inf
    patience_cnt  = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, soft_matrix, device)
        val_loss, val_acc = eval_epoch(model, val_loader, soft_matrix, device)
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

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
    test_loss, test_acc = eval_epoch(model, test_loader, soft_matrix, device)
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    eval_result = {"test_loss": test_loss, "test_acc": test_acc, "config": CONFIG}
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval()
    dummy = torch.zeros(1, CONFIG["input_dim"])
    torch.onnx.export(
        model, dummy,
        MODEL_DIR / "model.onnx",
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
