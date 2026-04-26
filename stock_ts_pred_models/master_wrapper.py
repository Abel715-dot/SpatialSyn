import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            qh = q[:, :, i * dim :] if i == self.nhead - 1 else q[:, :, i * dim : (i + 1) * dim]
            kh = k[:, :, i * dim :] if i == self.nhead - 1 else k[:, :, i * dim : (i + 1) * dim]
            vh = v[:, :, i * dim :] if i == self.nhead - 1 else v[:, :, i * dim : (i + 1) * dim]
            attn = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            attn = self.attn_dropout[i](attn)
            att_output.append(torch.matmul(attn, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class TAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            qh = q[:, :, i * dim :] if i == self.nhead - 1 else q[:, :, i * dim : (i + 1) * dim]
            kh = k[:, :, i * dim :] if i == self.nhead - 1 else k[:, :, i * dim : (i + 1) * dim]
            vh = v[:, :, i * dim :] if i == self.nhead - 1 else v[:, :, i * dim : (i + 1) * dim]
            attn = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            attn = self.attn_dropout[i](attn)
            att_output.append(torch.matmul(attn, vh))
        att_output = torch.concat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        out = self.trans(gate_input)
        out = torch.softmax(out / self.t, dim=-1)
        return self.d_output * out


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTERWrapper(nn.Module):
    """
    MASTER wrapper based on official implementation:
    https://github.com/SJTU-Quant/MASTER
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        d_feat: int = 5,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        t_dropout_rate: float = 0.5,
        s_dropout_rate: float = 0.5,
        gate_input_start_index: int = 5,
        gate_input_end_index: int = 10,
        beta: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        d_feat = int(kwargs.get("master_d_feat", d_feat))
        d_model = int(kwargs.get("master_d_model", d_model))
        t_nhead = int(kwargs.get("master_t_nhead", t_nhead))
        s_nhead = int(kwargs.get("master_s_nhead", s_nhead))
        t_dropout_rate = float(kwargs.get("master_t_dropout_rate", t_dropout_rate))
        s_dropout_rate = float(kwargs.get("master_s_dropout_rate", s_dropout_rate))
        gate_input_start_index = int(
            kwargs.get("master_gate_input_start_index", gate_input_start_index)
        )
        gate_input_end_index = int(
            kwargs.get("master_gate_input_end_index", gate_input_end_index)
        )
        beta = float(kwargs.get("master_beta", beta))

        self.task_type = task_type
        self.gate_input_start_index = int(gate_input_start_index)
        self.gate_input_end_index = int(gate_input_end_index)
        if self.gate_input_end_index > input_dim:
            self.gate_input_end_index = input_dim
        if self.gate_input_start_index >= self.gate_input_end_index:
            self.gate_input_start_index = max(0, input_dim - min(input_dim, 4))
            self.gate_input_end_index = input_dim
        self.d_feat = min(int(d_feat), self.gate_input_start_index)

        self.feature_gate = Gate(
            self.gate_input_end_index - self.gate_input_start_index,
            self.d_feat,
            beta=float(beta),
        )
        self.layers = nn.Sequential(
            nn.Linear(self.d_feat, int(d_model)),
            PositionalEncoding(int(d_model)),
            TAttention(d_model=int(d_model), nhead=int(t_nhead), dropout=float(t_dropout_rate)),
            SAttention(d_model=int(d_model), nhead=int(s_nhead), dropout=float(s_dropout_rate)),
            TemporalAttention(d_model=int(d_model)),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        src = x[:, :, : self.d_feat]
        gate_input = x[:, -1, self.gate_input_start_index : self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        return self.layers(src)

