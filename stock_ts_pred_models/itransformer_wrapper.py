import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class ITransformerWrapper(nn.Module):
    """
    iTransformer wrapper based on official implementation:
    https://github.com/thuml/iTransformer
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        seq_len: int = 15,
        target_feature_index: int = 1,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 256,
        e_layers: int = 2,
        factor: int = 3,
        dropout: float = 0.1,
        output_attention: bool = False,
        use_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.target_feature_index = int(target_feature_index)
        self.seq_len = int(seq_len)

        repo_root = Path(__file__).resolve().parents[1] / "third_party_models" / "iTransformer"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from model.iTransformer import Model as _OfficialITransformer

        cfg = SimpleNamespace(
            seq_len=self.seq_len,
            pred_len=1,
            output_attention=bool(output_attention),
            use_norm=bool(use_norm),
            d_model=int(d_model),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            factor=int(factor),
            dropout=float(dropout),
            embed="timeF",
            freq="h",
            class_strategy="projection",
            activation="gelu",
        )
        self.model = _OfficialITransformer(cfg)

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        if x.shape[1] < self.seq_len:
            pad_len = self.seq_len - x.shape[1]
            x = torch.cat([x[:, :1, :].expand(-1, pad_len, -1), x], dim=1)
        elif x.shape[1] > self.seq_len:
            x = x[:, -self.seq_len :, :]

        out = self.model(x, None, None, None)  # [B, 1, F]
        idx = min(max(self.target_feature_index, 0), out.shape[2] - 1)
        return out[:, -1, idx].unsqueeze(-1)

