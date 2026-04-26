import torch
import torch.nn as nn


class TimeAxisAttention(nn.Module):
    """
    DTML paper's time-axis attention block.
    Input: [D, W, L] -> Output context [D, H].
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(x)  # out: [D, W, H], h: [num_layers, D, H]
        h_last = h[-1].unsqueeze(-1)  # [D, H, 1]
        score = torch.bmm(out, h_last)  # [D, W, 1]
        attn = torch.softmax(score, dim=1)
        context = torch.bmm(attn.transpose(1, 2), out).squeeze(1)  # [D, H]
        return self.layer_norm(context)


class DataAxisAttention(nn.Module):
    """
    DTML paper's data-axis attention block.
    Input/Output: [D, H].
    """

    def __init__(self, hidden_size: int, num_heads: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention expects [B, N, E], so use B=1 and N=D.
        hm = h.unsqueeze(0)  # [1, D, H]
        attn_out, _ = self.multi_attn(hm, hm, hm, need_weights=False)
        hm_hat = self.layer_norm_1(hm + self.dropout(attn_out))
        ffn_out = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.layer_norm_2(hm_hat + self.dropout(ffn_out))
        return hp.squeeze(0)  # [D, H]


class DTMLWrapper(nn.Module):
    """
    DTML wrapper for this project.
    Input: x [B, T, F]
    Output: [B, 1]

    Default hyper-parameters align to QuantBench's DTML config:
    - out_features=256
    - num_heads=8
    - num_layers=1
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        node_emb_dim: int = 256,
        out_features: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        beta: float = 0.1,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.beta = float(beta)
        hidden_size = int(out_features)
        self.input_proj = nn.Linear(int(input_dim), int(node_emb_dim))
        self.time_attn = TimeAxisAttention(
            input_size=int(node_emb_dim),
            hidden_size=hidden_size,
            num_layers=int(num_layers),
        )
        self.data_attn = DataAxisAttention(
            hidden_size=hidden_size,
            num_heads=int(num_heads),
            drop_rate=float(dropout),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        # x: [B, T, F] where B is cross-sectional samples in a batch.
        # DTML uses a market-index sequence; here we use cross-sectional mean
        # as index proxy on each time step to preserve the architecture.
        x_proj = self.input_proj(x)  # [B, T, E]
        stocks = x_proj.permute(1, 0, 2)  # [W, D, E]
        index = stocks.mean(dim=1, keepdim=True)  # [W, 1, E]

        c_stocks = self.time_attn(stocks.transpose(1, 0))  # [D, H]
        c_index = self.time_attn(index.transpose(1, 0))  # [1, H]
        h_multi = c_stocks + self.beta * c_index.expand_as(c_stocks)
        h_refined = self.data_attn(h_multi)  # [D, H]
        return self.head(h_refined)  # [D, 1] == [B, 1]
