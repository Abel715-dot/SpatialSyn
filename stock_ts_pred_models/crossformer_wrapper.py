import sys
from pathlib import Path

import torch
import torch.nn as nn


class CrossformerWrapper(nn.Module):
    """
    Crossformer wrapper based on the official implementation:
    https://github.com/Thinklab-SJTU/Crossformer
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        seq_len: int = 15,
        target_feature_index: int = 1,
        seg_len: int = 6,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 256,
        d_ff: int = 512,
        n_heads: int = 4,
        e_layers: int = 3,
        dropout: float = 0.2,
        baseline: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.target_feature_index = int(target_feature_index)
        self.in_len = int(seq_len)

        repo_root = Path(__file__).resolve().parents[1] / "third_party_models" / "Crossformer"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from cross_models.cross_former import Crossformer as _OfficialCrossformer

        # ``device`` is only stored on the upstream module; parameters follow the wrapper after ``.to(device)``.
        self.model = _OfficialCrossformer(
            data_dim=int(input_dim),
            in_len=self.in_len,
            out_len=1,
            seg_len=int(seg_len),
            win_size=int(win_size),
            factor=int(factor),
            d_model=int(d_model),
            d_ff=int(d_ff),
            n_heads=int(n_heads),
            e_layers=int(e_layers),
            dropout=float(dropout),
            baseline=bool(baseline),
        )

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        if x.shape[1] < self.in_len:
            pad_len = self.in_len - x.shape[1]
            x = torch.cat([x[:, :1, :].expand(-1, pad_len, -1), x], dim=1)
        elif x.shape[1] > self.in_len:
            x = x[:, -self.in_len :, :]
        out = self.model(x)  # [B, 1, F]
        idx = min(max(self.target_feature_index, 0), out.shape[2] - 1)
        return out[:, -1, idx].unsqueeze(-1)

