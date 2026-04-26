import torch
import torch.nn as nn


class DeltaLagWrapper(nn.Module):
    """
    DeltaLag (Zhou et al., ICAIF 2025 / arXiv:2511.00390) cross-asset lead-lag model.

    Matches the paper §4.1–4.2 structure:
    - Temporal encoder f_theta on full window X_{u,t} (LSTM, as in the paper).
    - Query q_u from last-timestep embedding of target stock u.
    - Keys from each leader v over the last l_max *encoded* timesteps, with W^K.
    - Attention scores A_{u,v,l} = q_u · k_{v,l}; Top-K over (|S|-1) x l_max pairs.
    - Weights: softmax over the K selected logits only; this matches taking a full-row
      softmax after setting all non-top-K entries to -inf (diagonal already -inf).
    - Aggregate softmax-weighted *raw* leader features x_{v, t-τ} with τ = l_max - j.
    - MLP maps aggregated raw features to predicted return.

    Input x: [N, T, F] with N = same-day cross-section (use training_loss=mon/ic/mse+ day batch).
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str = "reg",
        *,
        hidden_size: int = 128,
        num_heads: int = 4,
        max_lag: int = 5,
        top_k: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = num_heads  # kept for CLI / sweep compatibility; not used in this architecture.
        max_lag = int(kwargs.get("deltalag_max_lag", max_lag))
        top_k = int(kwargs.get("deltalag_top_k", top_k))
        self.task_type = task_type
        self.hidden_size = int(hidden_size)
        self.max_lag = max(0, int(max_lag))
        # top_k==0 would break torch.topk; K>=1 is required for attention.
        self.top_k = max(1, int(top_k))
        self.input_dim = int(input_dim)

        # Paper: f_theta typically LSTM.
        self.temporal = nn.LSTM(
            self.input_dim,
            self.hidden_size,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # Paper §4.2: MLP on aggregated raw F-dimensional features.
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, x_sec=None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DeltaLag expects x shape [N,T,F], got {tuple(x.shape)}")
        n_stocks, seq_len, feat_dim = x.shape
        if feat_dim != self.input_dim:
            raise ValueError(
                f"DeltaLag input_dim mismatch: got F={feat_dim}, expected {self.input_dim}"
            )
        if n_stocks < 2:
            return self.head(x[:, -1, :])

        # Single temporal encoding per stock over the full window (paper Eq. 2–3).
        enc, _ = self.temporal(x)  # [N, T, H]
        q = self.query(enc[:, -1, :])  # [N, H] — last timestep embedding of each stock.

        l_max = min(self.max_lag, seq_len)
        # l_max==0 ⇒ no key axis; flat would be empty and top_idx//l_max would divide by zero.
        if l_max < 1:
            return self.head(x[:, -1, :])

        k_emb = enc[:, -l_max:, :]  # [N, l_max, H] — last l_max *encoded* steps (paper Eq. 6).
        k_bank = self.key(k_emb)  # [N, l_max, H]

        # scores[u, v, l] = q_u · k_{v,l}  (paper Eq. 7).
        scores = torch.einsum("uh,vlh->uvl", q, k_bank)  # [N, N, l_max]
        eye = torch.eye(n_stocks, dtype=torch.bool, device=x.device)
        scores = scores.masked_fill(eye.unsqueeze(-1), float("-inf"))

        flat = scores.reshape(n_stocks, -1)  # [N, N*l_max]; masked v=u entries are -inf
        # Never request more top entries than finite scores per row: (N-1)*l_max.
        n_finite = max((n_stocks - 1) * l_max, 1)
        k_top = min(self.top_k, flat.shape[1], n_finite)
        top_vals, top_idx = torch.topk(flat, k=k_top, dim=1)

        leader_idx = torch.div(top_idx, l_max, rounding_mode="floor")
        j = torch.remainder(top_idx, l_max)  # column index in K_{v,t}, paper j_m

        # τ = l_max - j (paper Eq. 11); raw index in window: (T-1) - τ = T - l_max + j - 1.
        raw_tidx = seq_len - l_max + j - 1
        raw_tidx = raw_tidx.clamp(0, seq_len - 1)

        z_leaders = x[leader_idx, raw_tidx, :]  # [N, k, F]

        w = torch.softmax(top_vals, dim=1).unsqueeze(-1)
        z_agg = (w * z_leaders).sum(dim=1)  # [N, F], paper Eq. 14
        return self.head(z_agg)
