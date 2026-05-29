"""
手牌類推モデル v5: Transformer self-attention across 34 tile tokens

入力 (219次元, v2/v4と同一):
  target_discard(44) + target_meld(38) + riichi(1) + score(11) + game(9) +
  self_discard(44) + self_meld(38) + visible_counts(34)

アーキテクチャ:
  1. global_encoder (219->256->d_model + BN): 全特徴量のグローバル圧縮
  2. tile_embed (Embedding[34, d_model]): タイル固有の学習表現
  3. visible_proj (Linear[1, d_model]): visible_count のタイル別情報
  4. TransformerEncoder (nhead=4, num_layers=3, Pre-LN): タイル間通信
  5. Per-tile head (Linear[d_model, 5]): 各タイルの枚数クラス予測

出力: (B, 34, 5)
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v5"

CONFIG = {
    "input_dim":    219,
    "d_model":      128,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "dropout":      0.1,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,
    "epochs":       60,
    "early_stop_patience": 7,
}

# visible_counts は219次元の末尾34次元 (index 185-218)
VISIBLE_OFFSET = 185


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


# ---- モデル (Transformer) ----

class HandInferenceV5(nn.Module):
    """
    34タイルをトークンとして扱い、self-attention で相互参照しながら枚数予測する。

    各タイルトークン = GlobalEncoder(219次元) + TileEmbed(tile_id) + VisibleProj(visible_count_i)
    """

    def __init__(self, input_dim, d_model, nhead, num_layers, n_pai, n_count_cls, dropout):
        super().__init__()
        self.n_pai = n_pai
        self.n_count_cls = n_count_cls

        # 全219次元をd_modelに圧縮 (BatchNorm1d で安定化)
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )

        # タイルID埋め込み [34, d_model]
        self.tile_embed = nn.Embedding(n_pai, d_model)

        # visible_count (スカラー1次元) → d_model
        self.visible_proj = nn.Linear(1, d_model, bias=False)

        # Transformer (Pre-LayerNorm, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-tile分類ヘッド
        self.head = nn.Linear(d_model, n_count_cls)

        # ONNX export時に torch.arange をトレース可能にするためbufferに登録
        self.register_buffer("tile_ids", torch.arange(n_pai))

    def forward(self, x):
        # x: [B, 219]
        g = self.global_encoder(x)            # [B, d_model]

        tile_emb = self.tile_embed(self.tile_ids)        # [34, d_model]

        # visible_counts: 末尾34次元を [B, 34, 1] に整形
        vis = x[:, VISIBLE_OFFSET:].unsqueeze(-1)         # [B, 34, 1]
        vis_emb = self.visible_proj(vis)                   # [B, 34, d_model]

        # タイルトークン: グローバル文脈 + タイルID + visible_count
        tokens = g.unsqueeze(1) + tile_emb.unsqueeze(0) + vis_emb  # [B, 34, d_model]

        out = self.transformer(tokens)         # [B, 34, d_model]
        return self.head(out)                  # [B, 34, 5]


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits.reshape(-1, model.n_count_cls), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits.reshape(-1, model.n_count_cls), labels.reshape(-1))
        total_loss += loss.item() * len(features)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
    return total_loss / len(loader.dataset), correct / total


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

    model = HandInferenceV5(
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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
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
    test_loss, test_acc = eval_epoch(model, test_loader, device)
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    eval_result = {"test_loss": test_loss, "test_acc": test_acc, "config": CONFIG}
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval()
    dummy = torch.zeros(1, CONFIG["input_dim"])
    torch.onnx.export(
        model, dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
