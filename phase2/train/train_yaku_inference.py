"""
役推定モデル v1 (Stage 1): 捨て牌・副露・リーチ → 役クラス多ラベル予測

入力 (92次元):
  target_discard(44) + target_meld(38) + riichi(1) + game_state(9)

出力: 11次元 binary (BCEWithLogitsLoss, multi-label)
  [tanyao, honitsu, chinitsu, yakuhai_haku, yakuhai_hatsu, yakuhai_chun,
   yakuhai_ba, yakuhai_ji, pinfu, chanta, riichi]

和了サンプル（won=True）のみで学習。
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "yaku_inference" / "v1"

CONFIG = {
    "input_dim":  108,
    "n_yaku":     21,
    "hidden":     [256, 128],
    "dropout":    0.2,
    "lr":         1e-3,
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "epochs":     40,
    "early_stop_patience": 7,
}


# ---- データセット ----

class YakuDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            [s["features"] for s in data], dtype=torch.float32
        )
        self.labels = torch.tensor(
            [s["label_yaku"] for s in data], dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---- モデル ----

class YakuInference(nn.Module):
    def __init__(self, input_dim, n_yaku, hidden, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_yaku))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits → apply sigmoid at inference time


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item() * len(features)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    # クラスごと精度 (threshold=0.5)
    per_class_acc = ((preds > 0.5) == labels.bool()).float().mean(0)
    macro_acc = per_class_acc.mean().item()
    return total_loss / len(loader.dataset), macro_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ndjson_path = DATA_DIR / "yaku_inference.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        print("extract_features.js を実行してから再試行してください")
        sys.exit(1)

    print(f"読み込み中: {ndjson_path}")
    with open(ndjson_path, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    print(f"総サンプル数: {len(all_data)}")

    # 和了サンプルのみ使用
    won_data = [s for s in all_data if s.get("won", False)]
    print(f"和了サンプル数: {len(won_data)} ({len(won_data)/len(all_data)*100:.1f}%)")

    sample_dim = len(won_data[0]["features"])
    if sample_dim != CONFIG["input_dim"]:
        print(f"次元数不一致: expected {CONFIG['input_dim']}, got {sample_dim}")
        sys.exit(1)

    random.seed(42)
    random.shuffle(won_data)
    n = len(won_data)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_data = won_data[:n_train]
    val_data   = won_data[n_train:n_train + n_val]
    test_data  = won_data[n_train + n_val:]
    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(YakuDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(YakuDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(YakuDataset(test_data),  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = YakuInference(
        input_dim = CONFIG["input_dim"],
        n_yaku    = CONFIG["n_yaku"],
        hidden    = CONFIG["hidden"],
        dropout   = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = math.inf
    patience_cnt  = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_macro_acc = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_acc={val_macro_acc:.4f}", flush=True)

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
    test_loss, test_macro_acc = eval_epoch(model, test_loader, device)
    print(f"test_loss={test_loss:.4f}  test_macro_acc={test_macro_acc:.4f}")

    eval_result = {"test_loss": test_loss, "test_macro_acc": test_macro_acc, "config": CONFIG}
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
