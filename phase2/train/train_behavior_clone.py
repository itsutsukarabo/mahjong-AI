"""
行動クローンモデル: 公開情報 + 自手牌から打牌を予測する

入力 (136次元):
  自手牌 (34)
  自捨て牌パターン (44)
  自副露パターン (38)
  点数状況 (11)
  ゲーム状況 (9)

出力:
  打牌する牌のインデックス (34クラス)
"""

import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- 設定 ----

DATA_DIR  = Path(__file__).parent.parent / "data" / "features"
MODEL_DIR = Path(__file__).parent.parent / "models" / "behavior_clone" / "v1"

CONFIG = {
    "input_dim":   136,
    "hidden_dims": [256, 256, 128],
    "n_actions":   34,
    "dropout":     0.3,
    "lr":          1e-3,
    "batch_size":  512,
    "epochs":      50,
    "early_stop_patience": 5,
}


# ---- データセット ----

class BehaviorCloneDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            [s["features"] for s in data], dtype=torch.float32
        )
        self.labels = torch.tensor(
            [s["label_action"] for s in data], dtype=torch.long
        ).clamp(0, CONFIG["n_actions"] - 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---- モデル ----

class BehaviorCloneModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_actions, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        total_loss += criterion(logits, labels).item() * len(features)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
    acc = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ndjson_path = DATA_DIR / "behavior_clone.ndjson"
    if not ndjson_path.exists():
        print(f"ファイルなし: {ndjson_path}")
        sys.exit(1)
    print(f"読み込み中: {ndjson_path}")
    with open(ndjson_path, encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    print(f"総サンプル数: {len(all_data)}")

    random.seed(42)
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_data = all_data[:n_train]
    val_data   = all_data[n_train:n_train + n_val]
    test_data  = all_data[n_train + n_val:]
    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    train_loader = DataLoader(BehaviorCloneDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(BehaviorCloneDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(BehaviorCloneDataset(test_data),  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = BehaviorCloneModel(
        input_dim   = CONFIG["input_dim"],
        hidden_dims = CONFIG["hidden_dims"],
        n_actions   = CONFIG["n_actions"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = math.inf
    patience_cnt  = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
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
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
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
