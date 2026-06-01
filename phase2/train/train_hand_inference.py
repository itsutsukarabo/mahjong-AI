"""
手牌類推モデル v6: v5 + global sum ソフト制約 + 4枚ハードマスク + 赤牌捨てシグナル
                      + 他家捨て牌 + 赤牌所持出力 + ポンスルー信号 + 役推定特徴量 + 聴牌確率

入力 (371次元):
  target_discard(44) + target_meld(38) + riichi(1) + score(11) + game(9) +
  self_discard(44)   + self_meld(38)   + visible_counts(34) +
  red_discard_signal(3)   [index 219:222] +
  other1_discard(44)      [index 222:266] +
  other2_discard(44)      [index 266:310] +
  pass_pon_signal(34)     [index 310:344] +
  wind(5)                 [index 344:349] +
  yaku_prob(21)           [index 349:370] +
  tenpai_prob(1)          [index 370:371]

アーキテクチャ (v5と同一):
  1. global_encoder (355->256->d_model + BN): 全特徴量のグローバル圧縮
  2. tile_embed (Embedding[34, d_model]): タイル固有の学習表現
  3. visible_proj (Linear[1, d_model]): visible_count のタイル別情報
  4. TransformerEncoder (nhead=4, num_layers=3, Pre-LN): タイル間通信
  5. Per-tile head (Linear[d_model, 5]): 各タイルの枚数クラス予測
  6. red_head (Linear[d_model, 2]): 5m/5p/5s の赤牌所持 binary 予測

損失関数:
  loss = CE + λ_gs * MSE(Σ E[count_i], Σ label_i)
       + λ_red_ce * CE(red_logits, label_red)
       + λ_red_cons * ReLU(P(赤持ち) - P(5を1枚以上持つ)).mean()

出力: logits(B, 34, 5), red_logits(B, 3, 2)
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
MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v6"

CONFIG = {
    "input_dim":    371,
    "d_model":      128,
    "nhead":        4,
    "num_layers":   3,
    "n_pai":        34,
    "n_count_cls":  5,
    "dropout":      0.1,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   1024,
    "epochs":       60,
    "early_stop_patience": 7,
}

VISIBLE_OFFSET     = 185
LAMBDA_GLOBAL_SUM  = 0.05
LAMBDA_RED_CE      = 0.3
LAMBDA_RED_CONS    = 0.1


# ---- データセット ----

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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.labels_red[idx]


# ---- モデル (Transformer + hard mask + red head) ----

class HandInferenceV6(nn.Module):
    """
    v5 Transformer + 4枚ハードマスク + 赤牌所持ヘッド。
    input_dim=355 (v5の219 + red_discard(3) + others(88) + pon_pass(34) + yaku(11))。

    各タイルトークン = GlobalEncoder(355次元) + TileEmbed(tile_id) + VisibleProj(visible_count_i)
    red_head: 5m(idx4)/5p(idx13)/5s(idx22) のトークンから binary 分類
    """

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

        self.tile_embed  = nn.Embedding(n_pai, d_model)
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
        self.head     = nn.Linear(d_model, n_count_cls)
        self.red_head = nn.Linear(d_model, 2)  # 赤牌所持 binary (2クラス)

        self.register_buffer("tile_ids",     torch.arange(n_pai))
        self.register_buffer("count_range",  torch.arange(n_count_cls))
        self.register_buffer("red_tile_idx", torch.tensor([4, 13, 22]))  # 5m/5p/5s

    def forward(self, x):
        # x: [B, input_dim]
        g = self.global_encoder(x)                                            # [B, d_model]

        tile_emb = self.tile_embed(self.tile_ids)                             # [34, d_model]
        vis = x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai].unsqueeze(-1)  # [B, 34, 1]
        vis_emb = self.visible_proj(vis)                                      # [B, 34, d_model]

        tokens = g.unsqueeze(1) + tile_emb.unsqueeze(0) + vis_emb             # [B, 34, d_model]
        out    = self.transformer(tokens)                                      # [B, 34, d_model]
        logits_raw = self.head(out)                                            # [B, 34, 5] マスク前

        # 4枚ハードマスク (推論用: visible_counts の二重計算バグでラベルがマスク領域に入るケースがあるため
        #                   学習時は logits_raw を使い、推論時のみ logits を使う)
        vis_raw = (x[:, VISIBLE_OFFSET:VISIBLE_OFFSET + self.n_pai] * 4).round().long().clamp(0, 4)
        hidden_remaining = (4 - vis_raw).clamp(min=0)                         # [B, 34]
        mask = self.count_range > hidden_remaining.unsqueeze(-1)              # [B, 34, 5]
        logits = logits_raw.masked_fill(mask, float('-inf'))

        # 赤牌予測: 5m/5p/5s のタイルトークンから binary 分類
        red_tokens = out[:, self.red_tile_idx, :]                             # [B, 3, d_model]
        red_logits = self.red_head(red_tokens)                                # [B, 3, 2]

        # logits=マスク済み(推論用), logits_raw=マスク前(学習用), red_logits
        return logits, logits_raw, red_logits


# ---- 学習・評価 ----

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_ce     = 0.0
    total_global = 0.0
    count_vals   = torch.arange(model.n_count_cls, device=device, dtype=torch.float32)
    five_idx     = model.red_tile_idx  # [3]

    for features, labels, labels_red in loader:
        features   = features.to(device)
        labels     = labels.to(device)
        labels_red = labels_red.to(device)
        optimizer.zero_grad()

        _, logits_raw, red_logits = model(features)                           # raw は学習用

        # CE損失はマスク前の raw logits で計算（visible_counts 二重計算バグ回避）
        loss_ce = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))

        # global sum ソフト制約 (raw logits)
        probs          = F.softmax(logits_raw, dim=-1)                        # [B, 34, 5]
        expected_total = (probs * count_vals).sum(-1).sum(-1)                 # [B]
        target_total   = labels.float().sum(-1)                               # [B]
        loss_global    = F.mse_loss(expected_total, target_total)

        # 赤牌 CE 損失
        loss_red_ce = F.cross_entropy(red_logits.reshape(-1, 2), labels_red.reshape(-1))

        # 整合性制約: P(赤持ち) ≤ P(5を1枚以上持つ) (raw logits)
        prob_has_red = F.softmax(red_logits, dim=-1)[:, :, 1]                # [B, 3]
        prob_cnt_ge1 = 1 - F.softmax(logits_raw[:, five_idx, :], dim=-1)[:, :, 0]  # [B, 3]
        loss_red_cons = F.relu(prob_has_red - prob_cnt_ge1).mean()

        loss = (loss_ce
                + LAMBDA_GLOBAL_SUM * loss_global
                + LAMBDA_RED_CE     * loss_red_ce
                + LAMBDA_RED_CONS   * loss_red_cons)
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
    for features, labels, labels_red in loader:
        features, labels = features.to(device), labels.to(device)
        logits, logits_raw, _ = model(features)
        # val_loss は raw logits (inf 回避), argmax は masked logits (推論動作に合わせる)
        loss = F.cross_entropy(logits_raw.reshape(-1, model.n_count_cls), labels.reshape(-1))
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
        print("add_yaku_features.py を実行してから再実行してください")
        sys.exit(1)

    if "label_red" not in all_data[0]:
        print("label_red フィールドがありません。extract_features.js を再実行してください")
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

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(HandInferenceDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    val_loader   = DataLoader(HandInferenceDataset(val_data),   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)
    test_loader  = DataLoader(HandInferenceDataset(test_data),  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=use_gpu, persistent_workers=True)

    model = HandInferenceV6(
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

    model.eval().cpu()
    dummy = torch.zeros(1, CONFIG["input_dim"])

    class _OnnxWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, red_logits = self.m(x)
            return logits, red_logits

    torch.onnx.export(
        _OnnxWrapper(model), dummy,
        str(MODEL_DIR / "model.onnx"),
        input_names=["features"],
        output_names=["logits", "red_logits"],
        dynamic_axes={
            "features":   {0: "batch_size"},
            "logits":     {0: "batch_size"},
            "red_logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"モデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
