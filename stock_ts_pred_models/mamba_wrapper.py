from typing import Optional

import torch
import torch.nn as nn


def _mamba_block_class(target_device: Optional[torch.device] = None):
    """
    Use fused ``mamba_ssm.Mamba`` only when training on CUDA and the package is installed.

    If ``target_device`` is ``mps`` or ``cpu`` (or CUDA is unavailable), use ``MambaTorch``
    so selective-scan does not call CUDA-only extensions — even on machines where CUDA exists.
    When ``target_device`` is None, fall back to "CUDA + import ok => fused" for backward compatibility.
    """
    want_fused = torch.cuda.is_available()
    if target_device is not None:
        want_fused = want_fused and target_device.type == "cuda"
    if want_fused:
        try:
            from mamba_ssm import Mamba

            return Mamba
        except ImportError:
            pass
    from .mamba_torch import MambaTorch

    return MambaTorch


class MambaWrapper(nn.Module):
    """
    Mamba single-step regression wrapper.
    Input: x [B, T, F], Output: [B, 1].

    Notes:
    - On CUDA with ``mamba-ssm`` installed, uses fused ``mamba_ssm.Mamba``.
    - Otherwise uses ``MambaTorch`` (pure PyTorch selective scan; supports MPS and CPU).
    - hidden_size is treated as model dimension (d_model).
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.d_model = int(hidden_size)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.dropout = nn.Dropout(dropout)

        target_dev = kwargs.pop("mamba_target_device", None)
        MambaBlock = _mamba_block_class(target_dev)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=self.d_model,
                    d_state=int(mamba_d_state),
                    d_conv=int(mamba_d_conv),
                    expand=int(mamba_expand),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        # x: [B, T, F]
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        h_last = self.dropout(h[:, -1, :])
        return self.head(h_last)
