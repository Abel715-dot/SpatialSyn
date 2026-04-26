# Wrapper: single sequence x [B,T,F] -> regression output [B,1]; uses Informer from paper repo.
import torch
import torch.nn as nn

from .informer_model import InformerModel


class InformerWrapper(nn.Module):
    """Informer 用于单步回归：输入 x [B,T,F]，输出 [B,1]。内部使用原论文实现（thuml/Autoformer 的 Informer）。"""
    def __init__(
        self,
        input_dim,
        task_type="reg",
        seq_len=20,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=3,
        dropout=0.1,
        embed="timeF",
        freq="h",
        activation="gelu",
        distil=True,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        self.pred_len = 1
        self.label_len = max(1, seq_len // 2)
        self._seq_len = seq_len
        configs = _InformerConfig(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=1,
            seq_len=seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=False,
            distil=distil,
        )
        self.core = InformerModel(configs)
        self._freq_d_inp = 4 if freq == "h" else 3  # timeF freq map

    def forward(self, x, x_sec=None):
        # x: [B, T, F]
        B, T, F = x.shape
        label_len = max(1, T // 2)
        pred_len = 1
        device = x.device
        dtype = x.dtype
        # 无时间特征时用零填充
        x_mark_enc = torch.zeros(B, T, self._freq_d_inp, device=device, dtype=dtype)
        x_dec = torch.cat([x[:, -label_len:, :], torch.zeros(B, pred_len, F, device=device, dtype=dtype)], dim=1)
        x_mark_dec = torch.zeros(B, label_len + pred_len, self._freq_d_inp, device=device, dtype=dtype)
        out = self.core(x, x_mark_enc, x_dec, x_mark_dec)
        return out.view(B, -1)[:, 0:1]


class _InformerConfig:
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        pred_len,
        d_model,
        n_heads,
        e_layers,
        d_layers,
        d_ff,
        factor,
        dropout,
        embed,
        freq,
        activation,
        output_attention,
        distil,
    ):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.output_attention = output_attention
        self.distil = distil
