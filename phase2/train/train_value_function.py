"""
価値関数モデル: 局面から局の得失点と最終着順点を予測する

入力 (67次元):
  自手牌 (34)
  点数状況 (11)
  ゲーム状況 (9)
  残り牌数 (1)
  他家3人の副露タイプ (12)

出力:
  label_round_fenpei: この局の得失点 (回帰)
  label_final_point:  最終着順点 (回帰)
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "value_function" / "v2"

CONFIG = {
    "input_dim":    67,
    "hidden_dims":  [128, 128, 64],
    "dropout":      0.3,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,
    "epochs":       50,
    "early_stop_patience": 5,
    "score_scale":  10000.0,
}


# ---- データセット ----

class ValueFunctionDataset(Dataset):
    def __init__(self, data, score_scale):
        self.features = torch.tensor(
            [s["features"] for s in data], dtype=torch.float32
        )
        self.label_round = torch.tensor(
            [s["label_round_fenpei"] / score_scale for s in data], dtype=torch.float32
        )
        self.label_final = torch.tensor(
            [s["label_final_point"] / score_scale for s in data], dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.label_round[idx], self.label_final[idx]


# ---- モデル ----

class ValueFunctionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_round = nn.Linear(prev, 1)
        self.head_final = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_round(h).squeeze(-1), self.head_final(h).squeeze(-1)


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for features, label_round, label_final in loader:
        features = features.to(device)
        label_round, label_final = label_round.to(device), label_final.to(device)
        optimizer.zero_grad()
        pred_round, pred_final = model(features)
        loss = criterion(pred_round, label_round) + criterion(pred_final, label_final)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, score_scale):
    model.eval()
    total_loss = 0.0
    mae_round = 0.0
    mae_final = 0.0
    criterion = nn.MSELoss()
    for features, label_round, label_final in loader:
        features = features.to(device)
        label_round, label_final = label_round.to(device), label_final.to(device)
        pred_round, pred_final = model(features)
        loss = criterion(pred_round, label_round) + criterion(pred_final, label_final)
        total_loss += loss.item() * len(features)
        mae_round += (pred_round - label_round).abs().sum().item() * score_scale
        mae_final += (pred_final - label_final).abs().sum().item() * score_scale
    n = len(loader.dataset)
    return total_loss / n, mae_round / n, mae_final / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ndjson_path = DATA_DIR / "value_function.ndjson"
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

    scale = CONFIG["score_scale"]
    train_loader = DataLoader(ValueFunctionDataset(train_data, scale), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ValueFunctionDataset(val_data,   scale), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(ValueFunctionDataset(test_data,  scale), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = ValueFunctionModel(
        input_dim   = CONFIG["input_dim"],
        hidden_dims = CONFIG["hidden_dims"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_loss = math.inf
    patience_cnt  = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae_round, val_mae_final = eval_epoch(model, val_loader, device, scale)
        scheduler.step(val_loss)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mae_round={val_mae_round:.0f}  mae_final={val_mae_final:.0f}")

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
    test_loss, test_mae_round, test_mae_final = eval_epoch(model, test_loader, device, scale)
    print(f"test_loss={test_loss:.4f}  mae_round={test_mae_round:.0f}  mae_final={test_mae_final:.0f}")

    eval_result = {
        "test_loss": test_loss,
        "test_mae_round_fenpei": test_mae_round,
        "test_mae_final_point":  test_mae_final,
        "config": CONFIG,
    }
    (MODEL_DIR / "eval_result.json").write_text(json.dumps(eval_result, indent=2))
    (MODEL_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2))

    model.eval()
    dummy = torch.zeros(1, CONFIG["input_dim"])
    torch.onnx.export(
        model, dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["pred_round", "pred_final"],
        dynamic_axes={"features": {0: "batch_size"}, "pred_round": {0: "batch_size"}, "pred_final": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
