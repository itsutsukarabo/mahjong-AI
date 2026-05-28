"""保存済み model.pt を ONNX にエクスポートする (v4用)"""
import json
import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "hand_inference" / "v4"

config = json.loads((MODEL_DIR / "config.json").read_text())


class HandInferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_pai, n_count_cls, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_pai * n_count_cls)
        self.n_pai       = n_pai
        self.n_count_cls = n_count_cls

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h).reshape(-1, self.n_pai, self.n_count_cls)


model = HandInferenceModel(
    input_dim   = config["input_dim"],
    hidden_dims = config["hidden_dims"],
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
