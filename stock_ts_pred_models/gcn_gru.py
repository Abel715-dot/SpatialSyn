# GCN-GRU 借自 Lucienxhh/GCN_GRU (Nature Scientific Reports, Visibility forecast based on GCN-GRU)
# https://github.com/Lucienxhh/GCN_GRU
# 适配单序列 (B,T,F)：将时间步视为节点，构建链式图，再 GCN -> GRU -> 单步输出。

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def _build_chain_edge_index(batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
    """为每个 batch 内的 seq_len 个节点构建链式边 (i<->i+1)，batch 内节点不跨图连接。"""
    edges = []
    for b in range(batch_size):
        offset = b * seq_len
        for i in range(seq_len - 1):
            edges.append([offset + i, offset + i + 1])
            edges.append([offset + i + 1, offset + i])
    if not edges:
        # seq_len==1: 自环
        edges = [[b * seq_len, b * seq_len] for b in range(batch_size)]
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


class GCNGRUWrapper(nn.Module):
    """
    GCN-GRU 单步回归：输入 x (B,T,F)。
    时间步作为图节点，链式图 + SAGEConv(GCN 风格) + GRU + Linear -> (B,1)。
    原实现: https://github.com/Lucienxhh/GCN_GRU (models.py 中 GCNGRU_Single)。
    """
    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if _HAS_PYG:
            self.conv1 = SAGEConv(input_dim, hidden_size)
            self.conv2 = SAGEConv(hidden_size, hidden_size)
        else:
            self.conv1 = None
            self.conv2 = None
            self._fallback_linear = nn.Linear(input_dim, hidden_size)
            self._fallback_conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        device = x.device
        if _HAS_PYG and self.conv1 is not None:
            node = x.reshape(-1, F)
            edge_index = _build_chain_edge_index(B, T, device)
            out = self.conv1(node, edge_index)
            out = torch.relu(out)
            out = self.conv2(out, edge_index)
            out = out.reshape(B, T, -1)
        else:
            # 无 PyG 时用 Linear + 1D 卷积模拟链上聚合
            out = torch.relu(self._fallback_linear(x))
            out = torch.relu(self._fallback_conv1d(out.transpose(1, 2)).transpose(1, 2))
        _, hidden = self.gru(out)
        last_h = hidden[-1]
        return self.head(last_h)
