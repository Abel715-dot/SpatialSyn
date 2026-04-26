import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


class PatchTSTWrapper(nn.Module):
    """
    PatchTST wrapper based on official implementation:
    https://github.com/yuqinie98/PatchTST
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        seq_len: int = 15,
        target_feature_index: int = 1,
        e_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 16,
        d_ff: int = 128,
        dropout: float = 0.3,
        patch_len: int = 16,
        stride: int = 8,
        fc_dropout: float = 0.3,
        head_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.target_feature_index = int(target_feature_index)
        self.seq_len = int(seq_len)

        repo_root = (
            Path(__file__).resolve().parents[1]
            / "third_party_models"
            / "PatchTST"
            / "PatchTST_supervised"
        )
        # iTransformer also defines a top-level `layers` package; clear cached
        # modules so PatchTST can load its own official layers package.
        for mod in list(sys.modules.keys()):
            if mod == "layers" or mod.startswith("layers."):
                del sys.modules[mod]
        sys.path = [
            p
            for p in sys.path
            if "/third_party_models/iTransformer" not in str(p)
        ]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from models.PatchTST import Model as _OfficialPatchTST

        cfg = SimpleNamespace(
            enc_in=int(input_dim),
            seq_len=self.seq_len,
            pred_len=1,
            e_layers=int(e_layers),
            n_heads=int(n_heads),
            d_model=int(d_model),
            d_ff=int(d_ff),
            dropout=float(dropout),
            fc_dropout=float(fc_dropout),
            head_dropout=float(head_dropout),
            individual=False,
            patch_len=int(patch_len),
            stride=int(stride),
            padding_patch="end",
            revin=True,
            affine=True,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.model = _OfficialPatchTST(cfg)

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        if x.shape[1] < self.seq_len:
            pad_len = self.seq_len - x.shape[1]
            x = torch.cat([x[:, :1, :].expand(-1, pad_len, -1), x], dim=1)
        elif x.shape[1] > self.seq_len:
            x = x[:, -self.seq_len :, :]

        out = self.model(x)  # [B, 1, F]
        idx = min(max(self.target_feature_index, 0), out.shape[2] - 1)
        return out[:, -1, idx].unsqueeze(-1)

