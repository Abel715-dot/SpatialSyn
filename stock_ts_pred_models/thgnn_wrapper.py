import math
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphAttnMultiHeadDense(nn.Module):
    """
    q4l GraphAttnMultiHead API with dense masked softmax.

    Upstream uses ``torch.sparse.softmax``, which is not implemented on MPS;
    this version matches the masked-graph semantics and runs on MPS/CUDA/CPU.
    """

    def __init__(
        self,
        in_features,
        out_features,
        negative_slope=0.2,
        num_heads=4,
        bias=True,
        residual=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(
            torch.FloatTensor(in_features, num_heads * out_features)
        )
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads * out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(
            dims=(1, 0, 2)
        )
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        e = self.leaky_relu(logits)
        neg_inf = torch.finfo(e.dtype).min
        mask = adj_mat.unsqueeze(0) > 0
        masked_e = torch.where(mask, e, neg_inf)
        attn_weights = torch.softmax(masked_e, dim=2)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(
            -1, self.num_heads * self.out_features
        )
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        return support, None


class THGNNWrapper(nn.Module):
    """
    THGNN embedded-dependency wrapper.
    Input: x [B, T, F], Output: [B, 1].

    Uses q4l's PairNorm / GraphAttnSemIndividual from:
      q4l.model.zoo.spatial.adaptive.thgnn
    Graph attention uses GraphAttnMultiHeadDense (dense softmax; MPS-compatible).

    To fit this project's interface, we build a dynamic positive/negative
    adjacency from in-batch stock correlations (on the latest Ts window),
    then apply q4l THGNN graph-attention blocks.
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        self.hidden_size = int(hidden_size)
        self.n_heads = max(1, int(n_heads))
        out_features = int(kwargs.get("out_features", self.hidden_size // self.n_heads))

        try:
            from q4l.model.zoo.spatial.adaptive.thgnn import (
                GraphAttnSemIndividual,
                PairNorm,
            )
        except Exception:
            # Some q4l environments cannot import full package because optional
            # compiled qlib extensions are missing. Fallback: load blocks from
            # q4l source (excluding GraphAttnMultiHead; we use GraphAttnMultiHeadDense).
            GraphAttnSemIndividual, PairNorm = self._load_blocks_from_q4l_source()

        self.temporal = nn.GRU(
            input_dim,
            self.hidden_size,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        if out_features <= 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) must be >= n_heads({self.n_heads}) for THGNN."
            )
        self.pos_gat = GraphAttnMultiHeadDense(
            in_features=self.hidden_size,
            out_features=out_features,
            num_heads=self.n_heads,
            residual=True,
        )
        self.neg_gat = GraphAttnMultiHeadDense(
            in_features=self.hidden_size,
            out_features=out_features,
            num_heads=self.n_heads,
            residual=True,
        )
        self.mlp_self = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp_pos = nn.Linear(out_features * self.n_heads, self.hidden_size)
        self.mlp_neg = nn.Linear(out_features * self.n_heads, self.hidden_size)
        self.pn = PairNorm(mode="PN-SI")
        self.sem_gat = GraphAttnSemIndividual(
            in_features=self.hidden_size,
            hidden_size=self.hidden_size,
            act=nn.Tanh(),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.hidden_size, 1)

    @staticmethod
    def _load_blocks_from_q4l_source():
        spec = importlib.util.find_spec("q4l")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(
                "THGNN embedded-dependency mode requires installed q4l package."
            )
        q4l_root = Path(list(spec.submodule_search_locations)[0])
        src_path = q4l_root / "model" / "zoo" / "spatial" / "adaptive" / "thgnn.py"
        if not src_path.exists():
            raise ImportError(f"Cannot find q4l THGNN source: {src_path}")
        code = src_path.read_text(encoding="utf-8")
        prefix = code.split("class THGNN", 1)[0]
        # Drop relative import that triggers heavy q4l package loading.
        prefix = prefix.replace("from .base import StockKG\n", "")
        # Only load reusable blocks; they do not depend on dgl runtime.
        prefix = prefix.replace("import dgl\n", "")
        prefix = prefix.replace("from dgl import DGLGraph\n", "")
        ns = {}
        exec(prefix, ns, ns)
        required = ("GraphAttnSemIndividual", "PairNorm")
        missing = [k for k in required if k not in ns]
        if missing:
            raise ImportError(
                f"Failed to load THGNN blocks from q4l source; missing: {missing}"
            )
        return ns["GraphAttnSemIndividual"], ns["PairNorm"]

    def _build_dynamic_adj(
        self, x: torch.Tensor, threshold: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        # Per-stock path proxy: mean over features per time step, then pairwise Pearson
        # (population std, same as former _corr). Vectorized O(B²) matmul on device — no Python loops.
        seq = x.mean(dim=2)  # [B, T]
        bsz = seq.shape[0]
        device = seq.device
        dtype = seq.dtype
        seq_c = seq - seq.mean(dim=1, keepdim=True)
        dots = seq_c @ seq_c.transpose(0, 1)
        norms_sq = (seq_c * seq_c).sum(dim=1).clamp(min=0.0)
        denom = (norms_sq.unsqueeze(1) * norms_sq.unsqueeze(0)).sqrt().clamp(min=1e-12)
        corr = torch.where(denom > 1e-12, dots / denom, torch.zeros((bsz, bsz), device=device, dtype=dtype))
        eye = torch.eye(bsz, device=device, dtype=dtype)
        off = ~torch.eye(bsz, dtype=torch.bool, device=device)
        ge = corr >= threshold
        pos_adj = eye + (off & ge).to(dtype)
        neg_adj = eye + (off & ~ge).to(dtype)
        return pos_adj, neg_adj

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        # x: [B, T, F], where B is typically one-day cross-section in IC training.
        out, _ = self.temporal(x)  # [B, T, H]
        h = out[:, -1, :]  # [B, H]
        pos_adj, neg_adj = self._build_dynamic_adj(x)

        pos_support, _ = self.pos_gat(h, pos_adj, requires_weight=False)
        neg_support, _ = self.neg_gat(h, neg_adj, requires_weight=False)

        self_emb = self.mlp_self(h)
        pos_emb = self.mlp_pos(pos_support)
        neg_emb = self.mlp_neg(neg_support)
        all_emb = torch.stack((self_emb, pos_emb, neg_emb), dim=1)  # [B, 3, H]
        all_emb, _ = self.sem_gat(all_emb, requires_weight=False)  # [B, H]
        all_emb = self.pn(all_emb)
        all_emb = self.dropout(all_emb)
        return self.head(all_emb)
