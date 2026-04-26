# MCI-GRU 官方结构：WinstonLiyt/MCI-GRU (Neurocomputing) code/csi300.py 中的 StockPredictionModel。
# 需与按 trade_date 的截面 batch 及 x_sec（图）一起使用；见 stock_ts_pred.train_stock_ts。
# https://github.com/WinstonLiyt/MCI-GRU

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch_geometric 在 GAT 层构造时惰性导入，避免未安装时整个 stock_ts_pred 无法 import。


# --- 以下模块与 third_party_models/MCI-GRU/code/csi300.py 保持一致 ---


def _require_gat_conv():
    try:
        from torch_geometric.nn import GATConv
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MCI-GRU 官方实现需要 torch_geometric（GATConv）。请安装: pip install torch_geometric"
        ) from e
    return GATConv


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.w_hh = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.attention = nn.Linear(hidden_size, input_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        attn_scores = self.attention(hidden)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = x * attn_weights

        gates = self.w_ih(x) + self.w_hh(hidden)
        r_gate, u_gate = gates.chunk(2, 2)

        r_gate = torch.sigmoid(r_gate)
        u_gate = torch.sigmoid(u_gate)

        h_hat = self.tanh(r_gate * hidden)
        new_hidden = u_gate * hidden + (1 - u_gate) * h_hat

        return new_hidden


class GATLayer(nn.Module):
    def __init__(self, hidden_size_gat1, output_gat1, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        GATConv = _require_gat_conv()
        self.gat1 = GATConv(in_channels, hidden_size_gat1, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat1 * heads, output_gat1, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


class GATLayer_1(nn.Module):
    def __init__(self, hidden_size_gat2, in_channels, out_channels, heads=1):
        super(GATLayer_1, self).__init__()
        GATConv = _require_gat_conv()
        self.gat1 = GATConv(in_channels, hidden_size_gat2, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_size_gat2 * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, query, key, value):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        k = k.transpose(-2, -1)

        attn_weights = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        k = k.transpose(-2, -1)

        attn_weights = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        return attn_output


class StockPredictionModel(nn.Module):
    """与官方 csi300.py 中 StockPredictionModel 同名同结构。"""

    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size_gat1,
        output_gat1,
        gat_in_channels,
        gat_out_channels,
        gat_heads,
        hidden_size_gat2,
        embed_dim,
        num_hidden_states,
    ):
        super(StockPredictionModel, self).__init__()
        _ = embed_dim  # 官方构造函数保留该参数，与仓库一致
        self.attention_gru = AttentionGRUCell(input_size, hidden_size)
        self.gat_layer = GATLayer(hidden_size_gat1, output_gat1, gat_in_channels, gat_out_channels, gat_heads)
        self.cross_attention = CrossAttention(hidden_size)
        self.num_hidden_states = num_hidden_states
        self.market_hidden_states_1 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.market_hidden_states_2 = nn.Parameter(torch.randn(num_hidden_states, hidden_size))
        self.self_attention = SelfAttention(hidden_size * 4)
        self.final_gat = GATLayer_1(hidden_size_gat2, hidden_size * 4, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x_time_series, x_graph, edge_index, edge_weight):
        batch_size, num_samples, num_time_steps, num_features = x_time_series.size()
        h_gru = torch.zeros(batch_size, num_samples, self.attention_gru.hidden_size).to(x_time_series.device)

        for t in range(num_time_steps):
            h_gru = self.attention_gru(x_time_series[:, :, t, :], h_gru)
        h_gru_1 = h_gru[-1, :, :]

        x_gat = self.gat_layer(x_graph, edge_index, edge_weight)

        stock_rep_1 = self.cross_attention(
            h_gru_1.unsqueeze(1), self.market_hidden_states_1, self.market_hidden_states_1
        ).squeeze(1)
        stock_rep_2 = self.cross_attention(
            x_gat.unsqueeze(1), self.market_hidden_states_2, self.market_hidden_states_2
        ).squeeze(1)

        concatenated_output = torch.cat([h_gru_1, x_gat, stock_rep_1, stock_rep_2], dim=1)

        attention_output = self.self_attention(concatenated_output.unsqueeze(1)).squeeze(1)

        out = self.final_gat(attention_output, edge_index, edge_weight)

        out = self.relu(out)

        return out.squeeze(1)


class MciGruGraphBuilder:
    """
    按官方 fun_graph：对相关系数矩阵，corr > judge 则连双向边，边权为相关系数。
    每个截面 batch 在子集股票上取子图（节点重编号 0..N-1）。
    """

    def __init__(self, ticker_order: Sequence[str], corr_matrix: np.ndarray, judge: float):
        self.ticker_order = list(ticker_order)
        self.judge = float(judge)
        self._t2i = {t: i for i, t in enumerate(self.ticker_order)}
        m = np.asarray(corr_matrix, dtype=np.float64)
        if m.shape != (len(self.ticker_order), len(self.ticker_order)):
            raise ValueError(
                f"corr_matrix shape {m.shape} != ({len(self.ticker_order)}, {len(self.ticker_order)})"
            )
        np.fill_diagonal(m, 0.0)
        self._M = m

    def edges_for_batch(self, codes_in_row_order: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = list(codes_in_row_order)
        n = len(codes)
        if n == 0:
            z = torch.zeros((2, 0), dtype=torch.long)
            w = torch.zeros((0,), dtype=torch.float32)
            return z, w

        gix = []
        for c in codes:
            if c not in self._t2i:
                raise KeyError(
                    f"MCI-GRU 图：股票 {c!r} 不在相关矩阵 universe 中（请检查面板与样本 ts_code 是否一致）"
                )
            gix.append(self._t2i[c])

        sub = self._M[np.ix_(gix, gix)]
        ei: List[List[int]] = []
        ew: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                wij = float(sub[i, j])
                if wij > self.judge:
                    ei.append([i, j])
                    ei.append([j, i])
                    ew.append(wij)
                    ew.append(wij)
        for i in range(n):
            ei.append([i, i])
            ew.append(1.0)
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(ew, dtype=torch.float32)
        return edge_index, edge_weight


def build_mci_gru_graph_builder_from_panel(
    panel: pd.DataFrame,
    *,
    tickers: Sequence[str],
    lookback: int,
    judge: float,
    return_col: str = "close_ret",
) -> MciGruGraphBuilder:
    """
    用最近 lookback 个交易日的截面收益估计相关矩阵（与官方 fun_relation 思路一致）。
    panel: load_panel 结果，需含 trade_date, ts_code, return_col。
    """
    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("build_mci_gru_graph_builder_from_panel: empty tickers")
    if return_col not in panel.columns:
        raise ValueError(f"panel 缺少列 {return_col!r}")
    sub = panel[panel["ts_code"].isin(tickers)].copy()
    sub = sub.sort_values(["trade_date", "ts_code"])
    dates = sub["trade_date"].unique()
    if len(dates) < max(5, lookback // 10):
        raise RuntimeError(
            f"MCI-GRU 构图：有效交易日过少 ({len(dates)})，无法稳定估计相关矩阵"
        )
    tail_dates = dates[-lookback:] if len(dates) >= lookback else dates
    w = sub[sub["trade_date"].isin(tail_dates)]
    pv = w.pivot(index="trade_date", columns="ts_code", values=return_col)
    pv = pv.reindex(columns=tickers).astype(np.float64)
    corr = pv.corr().reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 0.0)
    return MciGruGraphBuilder(tickers, corr, judge)


class MCIGRUWrapper(nn.Module):
    """
    官方 StockPredictionModel 封装。
    forward(x, x_sec):
      - x: (N, T, F) 同一 trade_date 的 N 只股票，序列长度 T，特征维 F（与官方 num_features 一致）。
      - x_sec: dict，键 x_graph (N,F)、edge_index (2,E)、edge_weight (E,)；与官方 model(x_ts, x, ei, ew) 一致。
    返回 (N, 1) 预测。
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        hidden_size: int = 32,
        hidden_size_gat1: int = 5,
        output_gat1: int = 32,
        gat_out_channels: int = 4,
        gat_heads: int = 4,
        hidden_size_gat2: int = 5,
        num_hidden_states: int = 16,
        embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        self.task_type = task_type
        self.core = StockPredictionModel(
            input_size=input_dim,
            hidden_size=hidden_size,
            hidden_size_gat1=hidden_size_gat1,
            output_gat1=output_gat1,
            gat_in_channels=input_dim,
            gat_out_channels=gat_out_channels,
            gat_heads=gat_heads,
            hidden_size_gat2=hidden_size_gat2,
            embed_dim=embed_dim,
            num_hidden_states=num_hidden_states,
        )

    def forward(self, x: torch.Tensor, x_sec: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if x_sec is None:
            raise ValueError(
                "MCI-GRU（官方实现）需要在每个截面 batch 上提供 x_sec："
                "x_graph (N,F)、edge_index、edge_weight；由 train_stock_ts 根据相关矩阵自动构造。"
            )
        if x.dim() != 3:
            raise ValueError(f"MCI-GRU 期望 x 形状 (N,T,F)，得到 {tuple(x.shape)}")
        x_ts = x.unsqueeze(0)
        x_graph = x_sec["x_graph"]
        ei = x_sec["edge_index"]
        ew = x_sec["edge_weight"]
        out = self.core(x_ts, x_graph, ei, ew)
        return out.view(-1, 1)
