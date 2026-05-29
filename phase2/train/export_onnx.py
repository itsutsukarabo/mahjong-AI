"""保存済み model.pt を ONNX にエクスポートする (v5用)"""
import json
import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v5"
VISIBLE_OFFSET = 185

config = json.loads((MODEL_DIR / "config.json").read_text())


class HandInferenceV5(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, n_pai, n_count_cls, dropout):
        super().__init__()
        self.n_pai = n_pai
        self.n_count_cls = n_count_cls
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.tile_embed = nn.Embedding(n_pai, d_model)
        self.visible_proj = nn.Linear(1, d_model, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_count_cls)
        self.register_buffer("tile_ids", torch.arange(n_pai))

    def forward(self, x):
        g = self.global_encoder(x)
        tile_emb = self.tile_embed(self.tile_ids)
        vis = x[:, VISIBLE_OFFSET:].unsqueeze(-1)
        vis_emb = self.visible_proj(vis)
        tokens = g.unsqueeze(1) + tile_emb.unsqueeze(0) + vis_emb
        out = self.transformer(tokens)
        return self.head(out)


model = HandInferenceV5(
    input_dim   = config["input_dim"],
    d_model     = config["d_model"],
    nhead       = config["nhead"],
    num_layers  = config["num_layers"],
    n_pai       = config["n_pai"],
    n_count_cls = config["n_count_cls"],
    dropout     = 0.0,
)
model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
model.eval()

dummy = torch.zeros(1, config["input_dim"])
torch.onnx.export(
    model, dummy,
    str(MODEL_DIR / "model.onnx"),
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17,
    dynamo=False,
)
print(f"ONNX export 完了: {MODEL_DIR / 'model.onnx'}")
