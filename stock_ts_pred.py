# encoding: utf-8
"""
基于 hs300 / sp500 数据源，滑动窗口构造 Xs(Ts,d)，对 corr>=thr 与 corr<=-thr 的相关股各做一套 PCA，
取最近 Ts 步并联拼入 (Ts, d+4*K+2)（正/负池主成分、解释方差、两池股票数）；可选再裁剪 ratio/corr 特征列，
用 LSTM/GRU/Informer/Autoformer 等预测未来 n 日收益 y = close_{t+n}/close_t - 1。
训练：在 [train_start, train_end] 上统一构建 panel/samples；trade_date < test_start 的样本按 80/20 划分 train/val（早停看 val）；
[test_start, test_end]（须落在上述区间内）为独立测试集，不再单独 load/build 测试段 panel。
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import re
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from stock_ts_pred_models.mci_gru import build_mci_gru_graph_builder_from_panel

from stock_ts_pred_models import (
    AutoformerWrapper,
    CrossformerWrapper,
    DeltaLagWrapper,
    DTMLWrapper,
    GCNGRUWrapper,
    InformerWrapper,
    ITransformerWrapper,
    MCIGRUWrapper,
    MambaWrapper,
    MASTERWrapper,
    PatchTSTWrapper,
    THGNNWrapper,
)
from stock_ts_pred_models.mamba_wrapper import _mamba_block_class


def get_db_path() -> str:
 
    return "./db"


def _cross_sectional_ic_bundle(y_true, y_pred, trade_dates) -> tuple[dict, pd.DataFrame]:
    """
    返回截面 IC 汇总 dict，以及按 trade_date 一行一日的 DataFrame（含 rank_ic / pearson_ic）。
    与 cross_sectional_ic_summary 的聚合口径一致；后者仅取 dict。
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    d = np.asarray(trade_dates).astype(str)
    if len(y) != len(p) or len(y) != len(d):
        raise ValueError("cross_sectional_ic_summary: y_true, y_pred, trade_dates 长度须一致")
    m = np.isfinite(y) & np.isfinite(p)
    y, p, d = y[m], p[m], d[m]
    empty = pd.DataFrame(columns=["trade_date", "rank_ic", "pearson_ic"])
    if len(y) < 2:
        return (
            {
                "rank_ic_mean": np.nan,
                "rank_ic_std": np.nan,
                "pearson_ic_mean": np.nan,
                "pearson_ic_std": np.nan,
                "n_days": 0,
                "n_samples": int(len(y)),
            },
            empty,
        )
    rank_ics, pearson_ics = [], []
    day_rows: List[dict] = []
    df = pd.DataFrame({"y": y, "p": p, "d": d})
    for day, g in df.groupby("d", sort=False):
        if len(g) < 2:
            continue
        yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
        if np.nanstd(yy) < 1e-12 or np.nanstd(pp) < 1e-12:
            continue
        rho_s = g["p"].corr(g["y"], method="spearman")
        rho_p = g["p"].corr(g["y"], method="pearson")
        day_rows.append(
            {"trade_date": str(day), "rank_ic": float(rho_s), "pearson_ic": float(rho_p)}
        )
        if np.isfinite(rho_s):
            rank_ics.append(float(rho_s))
        if np.isfinite(rho_p):
            pearson_ics.append(float(rho_p))
    rank_ics_arr = np.array(rank_ics, dtype=np.float64)
    pearson_ics_arr = np.array(pearson_ics, dtype=np.float64)
    r_mean = float(np.mean(rank_ics_arr)) if len(rank_ics_arr) else np.nan
    r_std = float(np.std(rank_ics_arr, ddof=1)) if len(rank_ics_arr) > 1 else np.nan
    p_mean = float(np.mean(pearson_ics_arr)) if len(pearson_ics_arr) else np.nan
    p_std = float(np.std(pearson_ics_arr, ddof=1)) if len(pearson_ics_arr) > 1 else np.nan
    out_df = pd.DataFrame(day_rows) if day_rows else empty
    return (
        {
            "rank_ic_mean": r_mean,
            "rank_ic_std": r_std,
            "pearson_ic_mean": p_mean,
            "pearson_ic_std": p_std,
            "n_days": int(len(rank_ics_arr)),
            "n_samples": int(len(y)),
        },
        out_df,
    )


def cross_sectional_ic_summary(y_true, y_pred, trade_dates) -> dict:
    """
    与常见量化研报一致的日度截面 IC：
    - Rank IC：每个 trade_date 上 Spearman(预测, 实现收益)，再对交易日取平均（秩相关，抗极端值）。
    - IC（截面 Pearson）：每个 trade_date 上 Pearson，再对交易日取平均。
    trade_dates 与 y_true/y_pred 逐行对齐（同一根样本的交易日）。
    """
    d, _ = _cross_sectional_ic_bundle(y_true, y_pred, trade_dates)
    return d


class ICLoss(nn.Module):
    """
    QuantBench / q4l 默认 ICLoss：在当前 batch 内对 pred 与 label 做整体 Pearson 相关，
    loss = -corr（最小化 loss 等价于最大化相关系数）。
    与 q4l.model.loss.zoo.ICLoss 一致；对常数向量或 batch 过小返回与 pred 连通的 0 标量，避免 NaN。
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dtype = pred.dtype
        p = pred.reshape(-1).to(dtype=dtype)
        y = label.reshape(-1).to(dtype=dtype)
        if p.numel() < 2:
            return (pred * 0).sum()
        p = p - p.mean()
        y = y - y.mean()
        vx = torch.sqrt(torch.sum(p * p))
        vy = torch.sqrt(torch.sum(y * y))
        if vx < self.eps or vy < self.eps:
            return (pred * 0).sum()
        cost = torch.sum(p * y) / (vx * vy)
        return -cost


class MonotonicRankingLoss(nn.Module):
    """
    DeltaLag paper-style monotonic logistic ranking loss:
      log(1 + exp(-tanh(p_i-p_j) * tanh(y_i-y_j))) over unordered pairs i<j
      (each unordered pair once; paper may sum i<j or double-count i!=j).
    Suitable for daily cross-sectional batches.
    """

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        p = pred.reshape(-1)
        y = label.reshape(-1)
        n = p.numel()
        if n < 2:
            return (pred * 0).sum()
        dp = p.unsqueeze(1) - p.unsqueeze(0)
        dy = y.unsqueeze(1) - y.unsqueeze(0)
        z = -torch.tanh(dp) * torch.tanh(dy)
        # Strict upper triangle: one term per unordered pair (i,j), i<j.
        mask = torch.triu(
            torch.ones(n, n, device=p.device, dtype=torch.bool), diagonal=1
        )
        pair = torch.log1p(torch.exp(z)) * mask
        return pair.sum() / mask.sum().clamp(min=1)


def _make_training_loss(name: str) -> nn.Module:
    key = name.strip().lower()
    if key in ("mse", "mse+"):
        return nn.MSELoss()
    if key in ("ic", "ic_loss", "icloss"):
        return ICLoss()
    if key in ("mon", "monotonic", "monotonic_ranking", "deltalag"):
        return MonotonicRankingLoss()
    if key in ("bce", "bce_logits", "bcewithlogits"):
        # Expect raw logits from model output; numerically stable sigmoid is inside the loss.
        return nn.BCEWithLogitsLoss()
    raise ValueError(
        f"Unknown training loss {name!r}; use 'mse', 'mse+', 'ic', 'mon' or 'bce'."
    )


def _daily_top_bottom_binary_labels(
    y_ret: np.ndarray,
    meta: List[Tuple[str, str]],
    *,
    top_k: int = 100,
    bottom_k: int = 100,
) -> np.ndarray:
    """
    Paper-style daily top/bottom-k labeling for BCE training.
    - top-k return samples => 1
    - bottom-k return samples => 0
    - middle samples => NaN (ignored in BCE training)
    """
    if top_k <= 0 and bottom_k <= 0:
        raise ValueError("For BCE labeling, at least one of top_k/bottom_k must be > 0.")
    labels = np.full(len(y_ret), np.nan, dtype=np.float32)
    by_day: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_day[str(m[1])].append(i)
    for _, idxs in by_day.items():
        if not idxs:
            continue
        arr = np.asarray([y_ret[i] for i in idxs], dtype=float)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        finite_local = np.where(finite)[0]
        arr_f = arr[finite]
        # Ascending order: first are worst (bottom), last are best (top).
        order = np.argsort(arr_f, kind="mergesort")
        if bottom_k > 0:
            for local_idx in order[: min(bottom_k, len(order))]:
                labels[idxs[finite_local[local_idx]]] = 0.0
        if top_k > 0:
            for local_idx in order[max(0, len(order) - top_k) :]:
                labels[idxs[finite_local[local_idx]]] = 1.0
    return labels


def _binary_labels_by_return_sign(y_ret: np.ndarray) -> np.ndarray:
    """
    DTML paper-style binary movement labeling:
    - return > 0 => 1
    - return < 0 => 0
    - return == 0 or non-finite => NaN (ignored)
    """
    arr = np.asarray(y_ret, dtype=float)
    labels = np.full(arr.shape[0], np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    pos = finite & (arr > 0)
    neg = finite & (arr < 0)
    labels[pos] = 1.0
    labels[neg] = 0.0
    return labels


def _daily_rank_labeling(y_ret: np.ndarray, meta: List[Tuple[str, str]]) -> np.ndarray:
    """
    Official MCI-GRU style labeling: for each trade_date, rank returns with pct=True.
    """
    arr = np.asarray(y_ret, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(meta):
        raise ValueError(
            f"_daily_rank_labeling: len(y_ret)={arr.shape[0]} != len(meta)={len(meta)}"
        )
    labels = np.full(arr.shape[0], np.nan, dtype=np.float32)
    by_day: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_day[str(m[1])].append(i)
    for _, idxs in by_day.items():
        if not idxs:
            continue
        day_vals = np.asarray([arr[i] for i in idxs], dtype=np.float32)
        ser = pd.Series(day_vals)
        # Align with official rank_labeling(...).rank(ascending=True, pct=True)
        ranked = ser.rank(ascending=True, pct=True).to_numpy(dtype=np.float32)
        labels[np.asarray(idxs, dtype=int)] = ranked
    return labels


def regression_report(
    y_true,
    y_pred,
    type_name="",
    trade_dates=None,
    *,
    daily_rank_ic_path: Optional[str] = None,
    model_type_label: Optional[str] = None,
) -> None:
    """原 train_model.regression_report：回归评估；有 trade_dates 时主报截面 Rank IC。"""
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    mae = (
        mean_absolute_error(y_true_arr[mask], y_pred_arr[mask]) if mask.sum() else float("nan")
    )
    if trade_dates is not None:
        # 只在需要落盘时构建逐日表；未开启 --daily_rank_ic_dir 时走轻量 cross_sectional_ic_summary，与改动前行为一致
        if daily_rank_ic_path and str(type_name).strip() == "Test":
            cs, daily_df = _cross_sectional_ic_bundle(y_true_arr, y_pred_arr, trade_dates)
            if daily_df is not None and not daily_df.empty:
                out = daily_df.copy()
                if model_type_label:
                    out.insert(0, "model_type", str(model_type_label))
                ddir = os.path.dirname(daily_rank_ic_path)
                if ddir:
                    os.makedirs(ddir, exist_ok=True)
                out.to_csv(daily_rank_ic_path, index=False)
                print(f"[daily_rank_ic] wrote: {daily_rank_ic_path} ({len(out)} rows)")
        else:
            cs = cross_sectional_ic_summary(y_true_arr, y_pred_arr, trade_dates)
        pooled_p = (
            pearsonr(y_true_arr[mask], y_pred_arr[mask])[0] if mask.sum() >= 2 else np.nan
        )
        line = (
            f"MAE={mae:.4f}, RankIC={cs['rank_ic_mean']:.4f}, "
            f"IC_pearson_daily={cs['pearson_ic_mean']:.4f}, "
            f"n_days={cs['n_days']}/{cs['n_samples']}, pooled_Pearson={pooled_p:.4f}"
        )
    else:
        pooled_p = (
            pearsonr(y_true_arr[mask], y_pred_arr[mask])[0] if mask.sum() >= 2 else np.nan
        )
        pooled_s = (
            pd.Series(y_true_arr[mask]).corr(pd.Series(y_pred_arr[mask]), method="spearman")
            if mask.sum() >= 2
            else np.nan
        )
        line = (
            f"MAE={mae:.4f}, pooled_Pearson={pooled_p:.4f}, pooled_Spearman={pooled_s:.4f} "
            f"(无 trade_date，非截面 IC)"
        )
    if type_name:
        print(f"{type_name}: {line}")
    else:
        print(line)


def set_seed(seed: int = 42) -> None:
    """原 time_model.set_seed：固定 random / numpy / torch / cudnn / PYTHONHASHSEED。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA / MPS: 统一先用 torch.manual_seed；只有 CUDA 才需要设置 cuda seed 与 cudnn 行为
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# 港股/英/日仍用 QuantBench 式 (open, close, impact) 比例；A 股/美股见分项模型。
BACKTEST_COST_LEGACY: Dict[str, Tuple[float, float, float]] = {
    "hk": (0.001, 0.001, 0.0),
    "uk": (0.0015, 0.0015, 0.0),
    "jp": (0.0015, 0.0015, 0.0),
}

# ---- A 股（人民币成交额）：佣金双向 + 单笔最低 5 元；印花税仅卖出 0.05%（2023-08 减半后万五）；过户费万 0.1 双向（约 0.001%）。
# 费率随券商/交易所细则可能略有差异，此处为常见零售口径。
DEFAULT_CN_COMMISSION_RATE = 0.00025  # 万 2.5
DEFAULT_CN_COMMISSION_MIN_CNY = 5.0
DEFAULT_CN_STAMP_DUTY_SELL = 0.0005  # 卖出单边
DEFAULT_CN_TRANSFER_FEE_RATE = 0.00001  # 万 0.1

# ---- 美股：线上佣金通常 0；卖出侧 SEC Section 31 费（按卖出金额比例，费率会调整，见 SEC 公告）；FINRA TAF 按股数、有上限。
# 无逐笔股数时用「参考股价」从市值推算股数，属必要近似。
DEFAULT_US_SEC_FEE_PER_DOLLAR_SOLD = 27.8e-6  # 约 $27.8 / $1M 卖出额（量级，以当期 SEC 费率为准）
DEFAULT_US_TAF_PER_SHARE = 0.000166
DEFAULT_US_TAF_CAP_PER_ORDER = 8.30
DEFAULT_US_REF_PRICE_PER_SHARE = 150.0  # 用于 TAF 估算，标普成分股可酌情调


@dataclass(frozen=True)
class BacktestCostSpec:
    """回测费用规格：none / legacy 比例 / cn 分项 / us 分项。"""

    kind: Literal["none", "legacy", "cn", "us"]
    market_key: str
    open_c: float = 0.0
    close_c: float = 0.0
    impact_c: float = 0.0
    cn_commission_rate: float = DEFAULT_CN_COMMISSION_RATE
    cn_commission_min_cny: float = DEFAULT_CN_COMMISSION_MIN_CNY
    cn_stamp_duty_sell: float = DEFAULT_CN_STAMP_DUTY_SELL
    cn_transfer_fee_rate: float = DEFAULT_CN_TRANSFER_FEE_RATE
    us_sec_fee_per_dollar_sold: float = DEFAULT_US_SEC_FEE_PER_DOLLAR_SOLD
    us_taf_per_share: float = DEFAULT_US_TAF_PER_SHARE
    us_taf_cap_per_order: float = DEFAULT_US_TAF_CAP_PER_ORDER
    us_ref_price_per_share: float = DEFAULT_US_REF_PRICE_PER_SHARE


def _make_backtest_cost_spec(
    preset: str,
    *,
    data_source: Optional[str] = None,
) -> BacktestCostSpec:
    raw = preset.strip().lower()
    key = raw
    if key == "auto":
        if data_source == "hs300":
            key = "cn"
        elif data_source == "sp500":
            key = "us"
        else:
            key = "none"
    if key == "none":
        return BacktestCostSpec(kind="none", market_key="none")
    if key == "cn":
        return BacktestCostSpec(kind="cn", market_key="cn")
    if key == "us":
        return BacktestCostSpec(kind="us", market_key="us")
    if key in BACKTEST_COST_LEGACY:
        o, c, i = BACKTEST_COST_LEGACY[key]
        return BacktestCostSpec(kind="legacy", market_key=key, open_c=o, close_c=c, impact_c=i)
    raise ValueError(
        f"Unknown backtest_cost_preset={preset!r}; "
        f"expected none, auto, cn, us, or one of {sorted(BACKTEST_COST_LEGACY.keys())}"
    )


def _cn_net_after_buy_gross(gross: float, r: float, t: float, min_c: float) -> float:
    comm = max(min_c, gross * r)
    return float(gross - comm - gross * t)


def _cn_gross_for_target_net(w_net: float, r: float, t: float, min_c: float) -> float:
    """买入：毛额 gross 满足 gross - max(min_c, gross*r) - gross*t = w_net（无印花税）。"""
    if w_net <= 0.0:
        return 0.0
    denom_full = 1.0 - r - t
    if denom_full > 1e-12:
        g1 = w_net / denom_full
        if g1 * r >= min_c - 1e-9:
            return float(g1)
    denom_part = 1.0 - t
    if denom_part <= 1e-12:
        return float(w_net + min_c)
    return float((w_net + min_c) / denom_part)


def _cn_equal_buy_net_allocations(
    cash_total: float,
    n: int,
    spec: BacktestCostSpec,
) -> List[float]:
    """将 cash_total 按 A 股买入费拆成 n 份等净买入；返回每只标的到账市值（含过户费、佣金）。"""
    if n <= 0 or cash_total <= 0.0:
        return []
    r, t, min_c = spec.cn_commission_rate, spec.cn_transfer_fee_rate, spec.cn_commission_min_cny

    def total_gross_for_wn(wn: float) -> float:
        return n * _cn_gross_for_target_net(wn, r, t, min_c)

    lo, hi = 0.0, cash_total
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        if total_gross_for_wn(mid) <= cash_total:
            lo = mid
        else:
            hi = mid
    w_net = lo
    g_each = _cn_gross_for_target_net(w_net, r, t, min_c)
    return [_cn_net_after_buy_gross(g_each, r, t, min_c) for _ in range(n)]


def _cn_sell_proceeds(notional: float, spec: BacktestCostSpec) -> float:
    """A 股卖出到账：佣金(max(5,·)) + 过户费 + 印花税（仅卖）。"""
    if notional <= 0.0:
        return 0.0
    w = float(notional)
    r, t, min_c = spec.cn_commission_rate, spec.cn_transfer_fee_rate, spec.cn_commission_min_cny
    comm = max(min_c, w * r)
    xfer = w * t
    stamp = w * spec.cn_stamp_duty_sell
    return float(w - comm - xfer - stamp)


def _us_sell_proceeds(notional: float, spec: BacktestCostSpec) -> float:
    if notional <= 0.0:
        return 0.0
    w = float(notional)
    sec = w * spec.us_sec_fee_per_dollar_sold
    px = spec.us_ref_price_per_share
    shares = w / px if px > 1e-9 else 0.0
    taf = min(spec.us_taf_cap_per_order, shares * spec.us_taf_per_share)
    return float(w - sec - taf)


def _backtest_cost_meta(spec: BacktestCostSpec) -> Dict[str, Any]:
    """写入回测结果 dict 的费用说明字段。"""
    out: Dict[str, Any] = {"cost_kind": spec.kind, "backtest_cost_market": spec.market_key}
    if spec.kind == "legacy":
        out["cost_open"] = spec.open_c
        out["cost_close"] = spec.close_c
        out["cost_impact"] = spec.impact_c
    elif spec.kind == "cn":
        out["cn_commission_rate"] = spec.cn_commission_rate
        out["cn_commission_min_cny"] = spec.cn_commission_min_cny
        out["cn_stamp_duty_sell"] = spec.cn_stamp_duty_sell
        out["cn_transfer_fee_rate"] = spec.cn_transfer_fee_rate
    elif spec.kind == "us":
        out["us_sec_fee_per_dollar_sold"] = spec.us_sec_fee_per_dollar_sold
        out["us_taf_per_share"] = spec.us_taf_per_share
        out["us_taf_cap_per_order"] = spec.us_taf_cap_per_order
        out["us_ref_price_per_share"] = spec.us_ref_price_per_share
    return out


def _optimal_desired_tickers(
    pred_ser: pd.Series,
    top_k: int,
    topk_pred_threshold: float,
) -> List[str]:
    """与原先逻辑一致：先阈值过滤，再按预测降序取前 top_k（可能少于 top_k）。"""
    p = pred_ser[pred_ser > float(topk_pred_threshold)].sort_values(ascending=False)
    return p.head(top_k).index.tolist()


def _desired_with_emax(
    held_set: set,
    pred_ser: pd.Series,
    top_k: int,
    topk_pred_threshold: float,
    emax: Optional[int],
) -> set:
    """
    在理想 top_k 集合基础上限制换手（Emax 为每轮上限）：
    - 无 Emax（None 或 <0）：目标组合 = 理想集 ideal（与原逻辑一致）。
    - 有 Emax≥0：从持仓中最多卖出 min(|held\\ideal|, Emax) 只（预测值在「应卖出池」中从低到高）；
      从 ideal\\held 中最多纳入 min(|ideal\\held|, Emax) 只（预测从高到低）。
    这样 ideal⊂held 时仍可逐步减仓；ideal 与 held 交错时每边各不超过 Emax 只调整（可净增/净减仓位数）。
    emax==0 表示本不调仓。若无任何标的通过阈值，返回空集（清仓）。
    """
    ideal_list = _optimal_desired_tickers(pred_ser, top_k, topk_pred_threshold)
    ideal: set = set(ideal_list)
    if not ideal:
        return set()
    if emax is None or emax < 0:
        return ideal
    held = set(held_set)
    if not held:
        return ideal
    emax_i = int(emax)
    if emax_i <= 0:
        return held
    out_pool = list(held - ideal)
    in_pool = list(ideal - held)
    if not out_pool and not in_pool:
        return held
    out_pool.sort(key=lambda s: float(pred_ser.get(s, float("-inf"))))
    in_pool.sort(key=lambda s: float(pred_ser.get(s, float("-inf"))), reverse=True)
    n_drop = min(len(out_pool), emax_i)
    n_add = min(len(in_pool), emax_i)
    drop = set(out_pool[:n_drop])
    add = set(in_pool[:n_add])
    return (held - drop) | add


def regression_as_class_report(y_true, y_pred, type_name="", threshold=0.0):
    """原 time_model.regression_as_class_report：按 threshold 二值化后算 Precision / Recall。"""
    y_true_cls = (np.array(y_true) > threshold).astype(int)
    y_pred_cls = (np.array(y_pred) > threshold).astype(int)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    prefix = f"{type_name} (as class): " if type_name else "(as class): "
    print(f"{prefix}Precision={precision:.4f}, Recall={recall:.4f}")


def _backtest_testset_cr_ar_sr_compare(
    *,
    meta_te: List[Tuple[str, str]],
    y_te: np.ndarray,
    test_pred: np.ndarray,
    n_forward: int,
    top_k: int = 20,
    topk_pred_threshold: float = 0.0,
    initial_cash: float = 1_000_000.0,
    annual_trading_days: int = 252,
    lite_info: bool = False,
    data_source: Optional[str] = None,
    backtest_cost_preset: Literal["none", "auto", "cn", "us", "hk", "uk", "jp"] = "none",
    rebalance_emax: Optional[int] = None,
) -> Dict[str, Any]:
    """
    仅使用“测试集”样本（trade_date 在 [test_start, test_end]）做两种策略对比：
    1) 基准：在测试集开始日把资金平均买入该阶段可用的全部股票并持有到结束；
    2) 优化：每隔 n_forward 个交易日先筛选 predict > topk_pred_threshold，再按预测从大到小选前 top_k；
       卖出掉出榜股票并用卖出资金买入新入榜股票（默认等权配置新进；已持有标的保留原市值权重）。

    可选费用：
    - backtest_cost_preset：none；auto（hs300→cn，sp500→us）；cn（A 股分项：双向佣金+最低5元、卖出印花税万五、过户费万0.1）；us（美股：卖出 SEC 比例费+FINRA TAF 按股估算）；hk/uk/jp 仍为 QuantBench 式比例费。
    - rebalance_emax：每轮再平衡对「应卖出池」最多卖 Emax 只、对「应买入池」最多纳入 Emax 只（两边独立计数；0 表示不调仓）；None 或 <0 表示不限制、目标直接为理想 top_k 集。
      该参数**仅用于含交易费用**的 Opt 回测；不含费用（ex-fee）的 Opt 固定按 unlimited 调仓，以便其 AR/SR 不受该参数影响。

    当 backtest_cost_preset 含交易费用时，会同时给出「不含交易费用」与「含交易费用」两套 CR/AR/SR（无费用路径与预设费用模型各跑一遍）。

    回测收益更新使用 y_te（其定义为未来 n_forward 日收益：close_{t+n}/close_t - 1）。
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if n_forward <= 0:
        raise ValueError(f"n_forward must be positive, got {n_forward}")
    if len(meta_te) == 0:
        print("Backtest skipped: empty test meta_te")
        return {}

    y_true_arr = np.asarray(y_te, dtype=np.float64).ravel()
    pred_arr = np.asarray(test_pred, dtype=np.float64).ravel()
    if len(y_true_arr) != len(pred_arr) or len(y_true_arr) != len(meta_te):
        raise ValueError(
            "Backtest input length mismatch: "
            f"len(meta_te)={len(meta_te)}, len(y_te)={len(y_true_arr)}, len(test_pred)={len(pred_arr)}"
        )

    trade_dates = [m[1] for m in meta_te]
    ts_codes = [m[0] for m in meta_te]
    df = pd.DataFrame(
        {"trade_date": trade_dates, "ts_code": ts_codes, "predict": pred_arr, "rev": y_true_arr}
    )
    df = df[np.isfinite(df["predict"]) & np.isfinite(df["rev"])].copy()
    if df.empty:
        print("Backtest skipped: no finite (predict, rev) rows in test set")
        return {}

    # 交易日排序（trade_date 为 YYYYMMDD 字符串时字典序一致）
    unique_dates = np.array(sorted(df["trade_date"].unique()), dtype=str)
    step = max(1, int(n_forward))  # “n_forward 日期后再次” -> 每隔 n_forward 个交易日再平衡
    # 从测试集首日开始作为首个再平衡日
    rebal_dates = unique_dates[::step]
    if len(rebal_dates) == 0:
        print("Backtest skipped: rebal_dates empty")
        return {}

    # 预先把每个再平衡日的预测与实现收益做成查找表（Series: index=ts_code）
    pred_by_date: Dict[str, pd.Series] = {}
    rev_by_date: Dict[str, pd.Series] = {}
    for d in rebal_dates:
        g = df[df["trade_date"] == d]
        if g.empty:
            continue
        pred_by_date[d] = g.set_index("ts_code")["predict"]
        rev_by_date[d] = g.set_index("ts_code")["rev"]

    valid_rebal_dates = [d for d in rebal_dates if d in pred_by_date and d in rev_by_date]
    if len(valid_rebal_dates) == 0:
        print("Backtest skipped: no valid rebal dates after filtering")
        return {}
    rebal_dates = valid_rebal_dates

    cost_spec = _make_backtest_cost_spec(backtest_cost_preset, data_source=data_source)
    cost_market = cost_spec.market_key
    use_cost = cost_spec.kind != "none"

    def _calc_metrics(values: List[float]) -> Dict[str, float]:
        # values: [t0_wealth, t1_wealth, ..., tM_wealth]，期间收益数量 M=len(values)-1
        if len(values) < 2:
            return {"CR": float("nan"), "AR": float("nan"), "SR": float("nan")}
        period_returns = np.asarray(values[1:], dtype=np.float64) / np.asarray(values[:-1], dtype=np.float64) - 1.0
        cr = float(values[-1] / values[0] - 1.0)
        m = len(period_returns)  # number of non-overlapping periods, each covers n_forward trading days
        total_days = m * n_forward
        if total_days <= 0:
            ar = float("nan")
        else:
            ar = float((values[-1] / values[0]) ** (annual_trading_days / total_days) - 1.0)
        if len(period_returns) < 2:
            sr = float("nan")
        else:
            denom = float(np.std(period_returns, ddof=1))
            if denom == 0.0 or not np.isfinite(denom):
                sr = float("nan")
            else:
                periods_per_year = float(annual_trading_days / n_forward)
                sr = float(np.sqrt(periods_per_year) * float(np.mean(period_returns)) / denom)
        return {"CR": cr, "AR": ar, "SR": sr}

    def _simulate(
        spec: BacktestCostSpec,
        *,
        silent: bool = False,
        rebalance_emax_effective: Optional[int],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        use_c = spec.kind != "none"
        buy_net = 1.0 - spec.open_c - spec.impact_c
        sell_net = 1.0 - spec.close_c - spec.impact_c
        if spec.kind == "legacy" and use_c and (buy_net <= 0 or sell_net <= 0):
            raise ValueError(
                f"Legacy transaction costs too large: buy_net={buy_net}, sell_net={sell_net} "
                f"(open={spec.open_c}, close={spec.close_c}, impact={spec.impact_c})"
            )

        def _legacy_init_equity() -> float:
            return float(initial_cash * buy_net) if use_c else float(initial_cash)

        # -------------------- 基准策略：全部股票等权持有 --------------------
        # 持有滚动需要每个再平衡日都有 rev，因此对“可用股票集合”取交集更严谨。
        base_universe = None
        for d in rebal_dates:
            codes_d = set(rev_by_date[d].index.tolist())
            base_universe = codes_d if base_universe is None else (base_universe & codes_d)
        base_codes = sorted(base_universe) if base_universe else []

        if len(base_codes) == 0:
            if not silent:
                print("Backtest warning: base universe empty (no stock has full rev coverage across rebal dates)")
            base_result: Dict[str, Any] = {}
        else:
            if spec.kind == "cn":
                nets_b = _cn_equal_buy_net_allocations(initial_cash, len(base_codes), spec)
                holdings = {s: float(nets_b[i]) for i, s in enumerate(base_codes)}
                init_eq_b = float(sum(nets_b))
            else:
                init_eq_b = (
                    float(initial_cash)
                    if spec.kind in ("none", "us")
                    else _legacy_init_equity()
                )
                wealth_each = init_eq_b / len(base_codes)
                holdings = {s: wealth_each for s in base_codes}  # 记录每只股票的当前财富（不追踪份额）
            cash = 0.0
            values = [init_eq_b]
            for d in rebal_dates:
                rev_ser = rev_by_date[d]
                for s in list(holdings.keys()):
                    # base_codes 已保证 rev 覆盖，缺失时按 0 处理更稳健
                    r = float(rev_ser.get(s, 0.0))
                    holdings[s] *= (1.0 + r)
                values.append(float(cash + sum(holdings.values())))

            base_metrics = _calc_metrics(values)
            base_result = {
                **base_metrics,
                "final_value": float(values[-1]),
                "n_periods": len(values) - 1,
                "n_stocks": len(base_codes),
                "backtest_cost_preset": backtest_cost_preset,
                **_backtest_cost_meta(spec),
            }

        # -------------------- 优化策略：top_k 选股 + 保持/替换 --------------------
        d0 = rebal_dates[0]
        pred0 = pred_by_date[d0]
        top0 = _optimal_desired_tickers(pred0, top_k, topk_pred_threshold)
        if len(top0) == 0:
            if not silent:
                print("Backtest warning: optimization top_k empty on start rebalance date")
            opt_result: Dict[str, Any] = {}
        else:
            if spec.kind == "cn":
                nets_o = _cn_equal_buy_net_allocations(initial_cash, len(top0), spec)
                holdings = {s: float(nets_o[i]) for i, s in enumerate(top0)}
                init_eq_o = float(sum(nets_o))
            else:
                init_eq_o = (
                    float(initial_cash)
                    if spec.kind in ("none", "us")
                    else _legacy_init_equity()
                )
                w0 = init_eq_o / len(top0)
                holdings = {s: float(w0) for s in top0}
            cash = 0.0
            held_set = set(top0)
            values = [init_eq_o]

            for k, d in enumerate(rebal_dates):
                rev_ser = rev_by_date[d]
                # 该再平衡日到下一个再平衡日前的实现收益滚动
                for s in list(holdings.keys()):
                    r = float(rev_ser.get(s, 0.0))
                    holdings[s] *= (1.0 + r)
                total_wealth = float(cash + sum(holdings.values()))
                values.append(total_wealth)

                # 最后一个 period 结束后不再调仓
                if k == len(rebal_dates) - 1:
                    break

                d_next = rebal_dates[k + 1]
                pred_next = pred_by_date[d_next]
                desired = _desired_with_emax(
                    held_set, pred_next, top_k, topk_pred_threshold, rebalance_emax_effective
                )

                continuing = held_set & desired
                dropped = held_set - continuing

                # 卖出掉出榜股票 -> 转为现金（分项或 legacy 比例）
                if dropped:
                    for s in dropped:
                        w = float(holdings.pop(s, 0.0))
                        if spec.kind == "cn":
                            cash += _cn_sell_proceeds(w, spec)
                        elif spec.kind == "us":
                            cash += _us_sell_proceeds(w, spec)
                        else:
                            cash += w * (sell_net if use_c else 1.0)

                # 新入榜 -> 用现金买入（分项或 legacy）
                new = desired - continuing
                if new and cash > 0.0:
                    if spec.kind == "cn":
                        nets_n = _cn_equal_buy_net_allocations(cash, len(new), spec)
                        cash = 0.0
                        for s, nx in zip(sorted(new), nets_n):
                            holdings[s] = float(nx)
                    elif spec.kind == "us":
                        deploy = cash
                        cash = 0.0
                        each = deploy / len(new)
                        for s in new:
                            holdings[s] = float(each)
                    else:
                        deploy = cash * (buy_net if use_c else 1.0)
                        cash = 0.0
                        wealth_each = deploy / len(new)
                        for s in new:
                            holdings[s] = float(wealth_each)
                elif new and cash <= 0.0:
                    for s in new:
                        holdings[s] = 0.0

                held_set = desired

            opt_metrics = _calc_metrics(values)
            opt_result = {
                **opt_metrics,
                "final_value": float(values[-1]),
                "n_periods": len(values) - 1,
                "top_k": top_k,
                "topk_pred_threshold": float(topk_pred_threshold),
                "rebalance_emax": rebalance_emax_effective,
                "backtest_cost_preset": backtest_cost_preset,
                **_backtest_cost_meta(spec),
            }

        return base_result, opt_result

    none_spec = BacktestCostSpec(kind="none", market_key="none")
    # 不含费用（ex-fee）路径固定 unlimited 调仓，不受 rebalance_emax 影响。
    base_ex, opt_ex = _simulate(none_spec, rebalance_emax_effective=None)
    if use_cost:
        # 含费用路径使用 rebalance_emax（若给定）。
        base_net, opt_net = _simulate(cost_spec, silent=True, rebalance_emax_effective=rebalance_emax)
    else:
        base_net, opt_net = base_ex, opt_ex

    def _preset_disp_str() -> str:
        return (
            f"{backtest_cost_preset}→{cost_market}"
            if str(backtest_cost_preset).strip().lower() == "auto"
            else str(backtest_cost_preset)
        )

    def _cost_detail_line(cs: BacktestCostSpec) -> str:
        _preset_disp = _preset_disp_str()
        if cs.kind == "none":
            return f"preset={_preset_disp} (no fees)"
        if cs.kind == "cn":
            return (
                f"preset={_preset_disp} A-share: comm max({cs.cn_commission_min_cny}CNY, "
                f"{cs.cn_commission_rate*100:.4f}%×V)/leg; stamp(sell) {cs.cn_stamp_duty_sell*100:.3f}%; "
                f"transfer {cs.cn_transfer_fee_rate*100:.4f}%/leg"
            )
        if cs.kind == "us":
            _usd_per_m = cs.us_sec_fee_per_dollar_sold * 1_000_000.0
            return (
                f"preset={_preset_disp} US: SEC(sell) ~${_usd_per_m:.2f}/$1M; "
                f"TAF ${cs.us_taf_per_share}/sh cap ${cs.us_taf_cap_per_order}; ref_px={cs.us_ref_price_per_share}"
            )
        return f"preset={_preset_disp} legacy open={cs.open_c}, close={cs.close_c}, impact={cs.impact_c}"

    def _print_base_opt_block(
        base_result: Dict[str, Any],
        opt_result: Dict[str, Any],
        *,
        emit_opt_empty_hint: bool = True,
    ) -> None:
        if base_result:
            print(
                "Base  | "
                f"CR={base_result.get('CR', float('nan')):.6f}, "
                f"AR={base_result.get('AR', float('nan')):.6f}, "
                f"SR={base_result.get('SR', float('nan')):.6f} | "
                f"final={base_result.get('final_value', float('nan')):.2f}, "
                f"n_stocks={base_result.get('n_stocks', 0)}"
            )
        if opt_result:
            _em = opt_result.get("rebalance_emax")
            _em_s = "∞" if _em is None or (isinstance(_em, (int, float)) and _em < 0) else str(int(_em))
            print(
                "Opt   | "
                f"CR={opt_result.get('CR', float('nan')):.6f}, "
                f"AR={opt_result.get('AR', float('nan')):.6f}, "
                f"SR={opt_result.get('SR', float('nan')):.6f} | "
                f"final={opt_result.get('final_value', float('nan')):.2f}, "
                f"top_k={opt_result.get('top_k', top_k)}, "
                f"Emax={_em_s}, "
                f"pred_thr={opt_result.get('topk_pred_threshold', topk_pred_threshold):.6f}"
            )
        elif base_result and emit_opt_empty_hint:
            print(
                "Opt   | (no metrics) 首个再平衡日 "
                f"{rebal_dates[0]} 上 predict>topk_pred_threshold({topk_pred_threshold}) 的股票数为 0，"
                "无法建仓；可调低 --topk_pred_threshold 或检查该日预测是否全为非正。"
            )

    print("")
    if not lite_info:
        print("Backtest on Test set: Base vs Opt")
        _emax_fee_s = "unlimited" if rebalance_emax is None or rebalance_emax < 0 else str(int(rebalance_emax))
        _emax_ex_s = "unlimited"
        print(
            f"Rebalance every {step} trade days (= n_forward={n_forward}) | "
            f"rebal_dates={rebal_dates[0]}..{rebal_dates[-1]} | n_periods={len(rebal_dates)} | "
            f"Emax(ex-fee)={_emax_ex_s}, Emax(fee)={_emax_fee_s}"
        )
        if use_cost:
            print("【不含交易费用】CR/AR/SR 不计佣金、印花税、冲击等")
            _print_base_opt_block(base_ex, opt_ex)
            print("【含交易费用】" + _cost_detail_line(cost_spec))
            _print_base_opt_block(base_net, opt_net, emit_opt_empty_hint=False)
        else:
            print(_cost_detail_line(none_spec))
            _print_base_opt_block(base_ex, opt_ex)
    else:
        if use_cost:
            print("【不含交易费用】")
            _print_base_opt_block(base_ex, opt_ex)
            print("【含交易费用】" + _cost_detail_line(cost_spec))
            _print_base_opt_block(base_net, opt_net, emit_opt_empty_hint=False)
        else:
            _print_base_opt_block(base_ex, opt_ex)

    out: Dict[str, Any] = {
        "base": base_net,
        "opt": opt_net,
        "base_ex_fee": base_ex,
        "opt_ex_fee": opt_ex,
    }
    return out


# Panel 缓存目录（仅与 data_source、start、end 绑定，Ts/Tw/n_forward 等为分片参数）
STOCK_TS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_ts_pred_cache")
SAMPLES_CACHE_VERSION = "v12_samples_npz_pcaLf_0_1_2_coef"


def _panel_cache_key(data_source: str, start: str, end: str) -> str:
    """仅用 data_source + 日期范围生成 key，panel 决定每只股票的大时间序列，Ts/Tw/n_forward 等为分片参数。"""
    raw = f"{data_source}|{start}|{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_cached_panel(cache_key: str) -> Optional[pd.DataFrame]:
    """返回缓存的 panel DataFrame 或 None。旧格式（样本元组）视为缓存失效。"""
    path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.pkl")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj
        return None  # 旧格式样本缓存，忽略
    except Exception:
        return None


def _save_cached_panel(cache_key: str, panel: pd.DataFrame) -> None:
    os.makedirs(STOCK_TS_CACHE_DIR, exist_ok=True)
    path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(panel, f)


def _samples_cache_key(
    *,
    data_source: str,
    start: str,
    end: str,
    Ts: int,
    Tw: int,
    n_forward: int,
    corr_threshold: float,
    corr_neg_threshold: float,
    min_corr_stocks: int,
    pca_components: int,
    features: Optional[List[str]] = None,
    pca_loading_sign_fix: int = 0,
) -> str:
    """
    二级缓存：缓存 build_samples 的输出 (X,y_by_forward,valid_by_forward,meta)。
    只与“样本构造”有关的参数绑定；与模型/训练超参无关，便于调参复用。
    正/负池开关 pos_sse、neg_sse 不参与 key（始终存完整列，读入后再按开关拼接）。
    pca_loading_sign_fix：0/1/2 均写入 key（|pcaLf=…），与定号算法一致。
    """
    feats = ",".join(features) if features else ""
    _plsf = int(pca_loading_sign_fix)
    if _plsf not in (0, 1, 2):
        raise ValueError(f"pca_loading_sign_fix must be 0, 1, or 2, got {_plsf}")
    pca_sf = f"|pcaLf={_plsf}"
    # 统一把与样本可用性/过滤相关的参数纳入 key，避免 plain 模式下 drop 发生错复用。
    raw = (
        f"{SAMPLES_CACHE_VERSION}|{data_source}|{start}|{end}|"
        f"Ts={Ts}|Tw={Tw}|n=1to5|thr={corr_threshold}|thrN={corr_neg_threshold}|minc={min_corr_stocks}|"
        f"pc={pca_components}|feats={feats}{pca_sf}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_cached_samples(
    cache_key: str,
    require_fallback_mask: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, Any], np.ndarray]]:
    """返回 (X, y_by_forward, valid_by_forward, meta, stats, fallback_mask) 或 None。"""
    npz_path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.npz")
    meta_path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.meta.pkl")
    if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            X = z["X"]
            y_by_forward = z["y_by_forward"] if "y_by_forward" in z.files else None
            valid_by_forward = z["valid_by_forward"] if "valid_by_forward" in z.files else None
            # 兼容旧缓存：只有单个 y（按当时 n_forward 构造）
            y_legacy = z["y"] if "y" in z.files else None
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)
        # payload: {"meta": [...], "stats": {...}, "fallback_mask": np.ndarray}
        if isinstance(payload, dict) and "meta" in payload:
            meta = payload.get("meta", [])
            stats = payload.get("stats", {})
            fallback_mask = payload.get("fallback_mask", None)
        else:
            # 兼容早期写入的纯 meta 列表
            meta = payload
            stats = {}
            fallback_mask = None
        if y_by_forward is None or valid_by_forward is None:
            return None
        y_by_forward = np.asarray(y_by_forward, dtype=np.float32)
        valid_by_forward = np.asarray(valid_by_forward, dtype=np.uint8)
        if y_by_forward.ndim != 2 or valid_by_forward.ndim != 2:
            return None
        if y_by_forward.shape != valid_by_forward.shape or y_by_forward.shape[1] != 5:
            return None
        if y_by_forward.shape[0] != X.shape[0]:
            return None

        if fallback_mask is None:
            if require_fallback_mask:
                return None
            fallback_mask_arr = np.zeros((len(meta),), dtype=np.uint8)
        else:
            fallback_mask_arr = np.asarray(fallback_mask, dtype=np.uint8).reshape(-1)
            if fallback_mask_arr.shape[0] != len(meta):
                if require_fallback_mask:
                    return None
                fallback_mask_arr = np.zeros((len(meta),), dtype=np.uint8)
        return X, y_by_forward, valid_by_forward, meta, stats, fallback_mask_arr
    except Exception:
        return None


def _save_cached_samples(
    cache_key: str,
    X: np.ndarray,
    y_by_forward: np.ndarray,
    valid_by_forward: np.ndarray,
    meta: List[Tuple[str, str]],
    fallback_mask: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(STOCK_TS_CACHE_DIR, exist_ok=True)
    npz_path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.npz")
    meta_path = os.path.join(STOCK_TS_CACHE_DIR, f"{cache_key}.meta.pkl")
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32, copy=False),
        y_by_forward=np.asarray(y_by_forward, dtype=np.float32),
        valid_by_forward=np.asarray(valid_by_forward, dtype=np.uint8),
    )
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "meta": meta,
                "stats": stats or {},
                "fallback_mask": np.asarray(fallback_mask, dtype=np.uint8).reshape(-1),
            },
            f,
        )


def _is_spatialsyn_family(model_type: str) -> bool:
    mt = str(model_type).upper().replace("-", "_")
    return mt in ("SPATIALSYN", "SPATIALSYN_PRO", "SPATIALSYN_LN", "SPATIALSYN_PRO_LN")


def _is_mci_gru(model_type: str) -> bool:
    mt = str(model_type).upper().replace("-", "_")
    return mt in ("MCI_GRU", "MCIGRU")


def _shift_ymd(ymd: str, delta_days: int) -> str:
    """YYYYMMDD ± 自然日。"""
    s = str(ymd).replace("-", "").strip()
    d = datetime.strptime(s, "%Y%m%d")
    return (d + timedelta(days=int(delta_days))).strftime("%Y%m%d")


def _mci_gru_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return xs, ys, metas


def _forward_stock_ts_batch(
    model: nn.Module,
    batch: Tuple,
    device: torch.device,
    *,
    mci_graph: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """统一 MCI-GRU（截面+图）与其余模型的 batch 前向。"""
    if mci_graph is not None:
        xb, yb, _metas = batch
        xb = xb.to(device)
        yb_t = yb.to(device)
        if yb_t.dim() == 1:
            yb_t = yb_t.unsqueeze(1)
        codes = [m[0] for m in _metas]
        ei, ew = mci_graph.edges_for_batch(codes)
        ei, ew = ei.to(device), ew.to(device)
        x_graph = xb[:, -1, :].contiguous()
        out = model(xb, {"x_graph": x_graph, "edge_index": ei, "edge_weight": ew})
        return out, yb_t
    xb, yb = batch
    xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
    out = model(xb, None)
    return out, yb


def build_model(model_type: str, input_dim: int, task_type: str, **kwargs) -> nn.Module:
    """支持 LSTM/GRU/Informer/Autoformer/Mamba/THGNN/SpatialSyn/SpatialSyn_Pro/SpatialSyn_LN/SpatialSyn_Pro_LN/GCN-GRU/MCI-GRU/DTML/Crossformer/iTransformer/PatchTST/MASTER/DeltaLag。"""
    mt = model_type.upper()
    if mt == "LSTM":
        return LSTMModel(input_dim, task_type=task_type, **kwargs)
    if mt == "GRU":
        return GRUModel(input_dim, task_type=task_type, **kwargs)
    if mt == "INFORMER":
        kw = _kwargs_for_transformer(kwargs)
        return InformerWrapper(input_dim, task_type=task_type, **kw)
    if mt == "AUTOFORMER":
        kw = _kwargs_for_transformer(kwargs)
        return AutoformerWrapper(input_dim, task_type=task_type, **kw)
    if mt == "MAMBA":
        return MambaWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "THGNN":
        return THGNNWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "DTML":
        return DTMLWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "CROSSFORMER":
        return CrossformerWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "ITRANSFORMER":
        return ITransformerWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "PATCHTST":
        return PatchTSTWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "MASTER":
        return MASTERWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "DELTALAG":
        return DeltaLagWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "SPATIALSYN":
        return SpatialSynModel(input_dim, task_type=task_type, **kwargs)
    if mt == "SPATIALSYN_PRO":
        return SpatialSynProModel(input_dim, task_type=task_type, **kwargs)
    if mt == "SPATIALSYN_LN":
        return SpatialSynModel(input_dim, task_type=task_type, **kwargs)
    if mt == "SPATIALSYN_PRO_LN":
        return SpatialSynProModel(input_dim, task_type=task_type, **kwargs)
    if mt == "GCN-GRU" or mt == "GCNGRU":
        return GCNGRUWrapper(input_dim, task_type=task_type, **kwargs)
    if mt == "MCI-GRU" or mt == "MCIGRU":
        return MCIGRUWrapper(input_dim, task_type=task_type, **kwargs)
    raise ValueError(
        f"Unsupported model_type: {model_type}. "
        "Use LSTM / GRU / Informer / Autoformer / Mamba / THGNN / SpatialSyn / SpatialSyn_Pro / SpatialSyn_LN / SpatialSyn_Pro_LN / "
        "GCN-GRU / MCI-GRU / DTML / Crossformer / iTransformer / PatchTST / MASTER / DeltaLag."
    )


def _kwargs_for_transformer(kwargs: dict) -> dict:
    """统一 Informer/Autoformer 参数名：nhead->n_heads, t_layers->e_layers, ff_dim->d_ff。"""
    out = dict(kwargs)
    if "nhead" in out and "n_heads" not in out:
        out["n_heads"] = out.pop("nhead")
    if "t_layers" in out and "e_layers" not in out:
        out["e_layers"] = out.pop("t_layers")
    if "ff_dim" in out and "d_ff" not in out:
        out["d_ff"] = out.pop("ff_dim")
    return out


class LSTMModel(nn.Module):
    """LSTM 时序模型，接口与 GRUModel 一致。"""
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.1, task_type="reg", **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.task_type = task_type

    def forward(self, x, x_sec=None):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class SpatialSynModel(nn.Module):
    """
    双支路：个股特征与 PCA 主成分序列各走一层 LSTM；解释方差占比与 corr_count 在样本内沿 Ts 为常数，
    仅从序列末步取一次，与两支路最后时刻 hidden 拼接后过 MLP 回归。
    输入列布局与 train_stock_ts 在 use_corr_pca=True 时一致（K=pca_components；正/负池可按 pos_sse/neg_sse 省略）：
    [base_d | 正池PCA|负池PCA 共 pca_dim=K*(侧数) | ratio 各 K*(侧数)? | n_pos/n_neg 各 0~1 列?]。
    """

    def __init__(
        self,
        input_dim: int,
        *,
        base_dim: int,
        pca_dim: int,
        ratio_dim: int = 0,
        coef_dim: int = 0,
        corr_dim: int = 0,
        hidden_size: int = 64,
        pca_lstm_hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        task_type: str = "reg",
        use_fusion_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if base_dim + pca_dim + ratio_dim + coef_dim + corr_dim != input_dim:
            raise ValueError(
                f"SpatialSyn layout mismatch: base_dim={base_dim} + pca_dim={pca_dim} + "
                f"ratio_dim={ratio_dim} + coef_dim={coef_dim} + corr_dim={corr_dim} != input_dim={input_dim}"
            )
        self.base_dim = base_dim
        self.pca_dim = pca_dim
        self.ratio_dim = ratio_dim
        self.coef_dim = coef_dim
        self.corr_dim = corr_dim
        self.task_type = task_type
        drop = dropout if num_layers > 1 else 0.0
        self.lstm_stock = nn.LSTM(
            base_dim, hidden_size, num_layers=num_layers,
            dropout=drop, batch_first=True,
        )
        self.lstm_pca = nn.LSTM(
            pca_dim, pca_lstm_hidden_size, num_layers=num_layers,
            dropout=drop, batch_first=True,
        )
        tail = ratio_dim + coef_dim + corr_dim
        fused = hidden_size + pca_lstm_hidden_size + tail
        self.fusion_ln = nn.LayerNorm(fused) if use_fusion_layernorm else nn.Identity()
        hid = max(fused // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(fused, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, x, x_sec=None):
        b = self.base_dim
        p = self.pca_dim
        rq, kq, cq = self.ratio_dim, self.coef_dim, self.corr_dim
        x_stock = x[:, :, :b]
        x_pca = x[:, :, b : b + p]
        _, (h_s, _) = self.lstm_stock(x_stock)
        _, (h_p, _) = self.lstm_pca(x_pca)
        parts = [h_s[-1], h_p[-1]]
        off = b + p
        if rq > 0:
            parts.append(x[:, -1, off : off + rq])
            off += rq
        if kq > 0:
            parts.append(x[:, -1, off : off + kq])
            off += kq
        if cq > 0:
            parts.append(x[:, -1, off : off + cq])
        h = torch.cat(parts, dim=-1)
        h = self.fusion_ln(h)
        return self.head(h)


class SpatialSynProModel(nn.Module):
    """与 SpatialSynModel 相同双 LSTM 支路；区别在于两支路输出先相加，再与 tail 拼接后进入 MLP 头。"""

    def __init__(
        self,
        input_dim: int,
        *,
        base_dim: int,
        pca_dim: int,
        ratio_dim: int = 0,
        coef_dim: int = 0,
        corr_dim: int = 0,
        hidden_size: int = 64,
        pca_lstm_hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        task_type: str = "reg",
        use_fusion_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if base_dim + pca_dim + ratio_dim + coef_dim + corr_dim != input_dim:
            raise ValueError(
                f"SpatialSyn_Pro layout mismatch: base_dim={base_dim} + pca_dim={pca_dim} + "
                f"ratio_dim={ratio_dim} + coef_dim={coef_dim} + corr_dim={corr_dim} != input_dim={input_dim}"
            )
        self.base_dim = base_dim
        self.pca_dim = pca_dim
        self.ratio_dim = ratio_dim
        self.coef_dim = coef_dim
        self.corr_dim = corr_dim
        self.task_type = task_type

        d_stock = int(hidden_size)
        d_pca = int(pca_lstm_hidden_size if pca_lstm_hidden_size is not None else hidden_size)
        
        drop = dropout if num_layers > 1 else 0.0

        self.lstm_stock = nn.LSTM(
            base_dim, d_stock, num_layers=num_layers,
            dropout=drop, batch_first=True,
        )
        self.lstm_pca = nn.LSTM(
            pca_dim, d_pca, num_layers=num_layers,
            dropout=drop, batch_first=True,
        )

        self.ratio_proj = nn.Linear(ratio_dim, d_pca) if ratio_dim > 0 else None
        self.coef_proj = nn.Linear(coef_dim, d_pca) if coef_dim > 0 else None
        self.corr_proj = nn.Linear(corr_dim, d_pca) if corr_dim > 0 else None
        fused = d_stock + d_pca
        self.fusion_ln = nn.LayerNorm(fused) if use_fusion_layernorm else nn.Identity()
        hid = d_stock #max(fused // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(fused, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, x, x_sec=None):
        b = self.base_dim
        p = self.pca_dim
        rq, kq, cq = self.ratio_dim, self.coef_dim, self.corr_dim
        x_stock = x[:, :, :b]
        x_pca = x[:, :, b : b + p]

        _, (h_s, _) = self.lstm_stock(x_stock)
        _, (h_p, _) = self.lstm_pca(x_pca)
        pca_feat = h_p[-1]
        off = b + p
        if rq > 0:
            pca_feat = pca_feat + self.ratio_proj(x[:, -1, off : off + rq])
            off += rq
        if kq > 0:
            pca_feat = pca_feat + self.coef_proj(x[:, -1, off : off + kq])
            off += kq
        if cq > 0:
            pca_feat = pca_feat + self.corr_proj(x[:, -1, off : off + cq])
        h = torch.cat([h_s[-1], pca_feat], dim=-1)
        h = self.fusion_ln(h)
        return self.head(h)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.1, task_type="reg", **kwargs):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.task_type = task_type

    def forward(self, x, x_sec=None):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

# --------------- 1. 数据加载：hs300 / sp500 全量面板 ---------------

def _panel_from_df_wide(df_wide: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """宽表 (date x stock) 转为长表 panel，仅用于 close 收益。需与多列面板合并。"""
    return df_wide.stack().reset_index().rename(columns={"level_0": "trade_date", "level_1": "ts_code", 0: "close_ret"})


def load_panel(
    start_date: str,
    end_date: str,
    data_source: Literal["hs300", "sp500"],
) -> pd.DataFrame:
    """
    加载指定区间的全量面板：ts_code, trade_date, open, high, low, close, pre_close, vol, turnover_rate_f，
    以及派生列 open_ret, close_ret, high_ret, low_ret（均为 /pre_close - 1）。
    data_source:
      'hs300' 从 train_db.hs_300_ts；
      'sp500' 从 train_db.sp500。
    """
    db_path = get_db_path()
    if data_source == "hs300":
        conn = sqlite3.connect(os.path.join(db_path, "train_db.db"))
        sql = """
            SELECT ts_code, trade_date, open, high, low, close, pre_close,
                   volume AS vol, turnover_rate_f
            FROM hs_300_ts
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date, ts_code
        """
        df = pd.read_sql(sql, conn, params=(start_date, end_date))
        conn.close()
    elif data_source == "sp500":
        conn = sqlite3.connect(os.path.join(db_path, "train_db.db"))
        sql = """
            SELECT ts_code, trade_date, open, high, low, close, pre_close,
                   volume AS vol, turnover_rate_f
            FROM sp500
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date, ts_code
        """
        df = pd.read_sql(sql, conn, params=(start_date, end_date))
        conn.close()
    else:
        raise ValueError(f"Unknown data_source={data_source!r}; expected 'hs300' or 'sp500'")

    df = df.groupby(["trade_date", "ts_code"], as_index=False).first()
    # 统一类型与排序，便于下游 build_samples 直接使用，减少重复 sort/copy
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    valid = pd.notna(df["pre_close"]) & (df["pre_close"] > 0)
    for col in ["open", "high", "low", "close"]:
        df.loc[valid, f"{col}_ret"] = df.loc[valid, col] / df.loc[valid, "pre_close"] - 1.0
    df.loc[~valid, "open_ret"] = df.loc[~valid, "close_ret"] = np.nan
    df.loc[~valid, "high_ret"] = df.loc[~valid, "low_ret"] = np.nan
    return df


# --------------- 2. 滑动窗口 + 相关股票 + PCA 构造 (Ts, d+pca 或 d+2pca) 与 y ---------------

# d = 6: open_ret, close_ret, high_ret, low_ret, vol, turnover_rate_f
DEFAULT_TS_FEATURE_NAMES = ["open_ret", "close_ret", "high_ret", "low_ret", "vol", "turnover_rate_f"]
# A股日常涨跌停约 ±10%，超过此阈值视为异常日（如上市首日无涨跌停），参与标准化会扭曲尺度
ABNORMAL_RET_THRESHOLD = 0.11
# Extra numeric guards for mixed data sources (e.g. sp500) to avoid PCA overflow.
RET_BLOCK_CLIP_ABS = 5.0
PCA_INPUT_CLIP_ABS = 20.0


def _pca_sign_should_flip_mode1(lk: np.ndarray) -> bool:
    """模式 1：列和为负则翻转；和为 0 时若 |载荷| 最大分量为负则翻转。"""
    sm = float(np.sum(lk))
    if sm < 0.0:
        return True
    if sm == 0.0 and lk.size > 0:
        j = int(np.argmax(np.abs(lk)))
        return float(lk[j]) < 0.0
    return False


def _align_pca_sign_from_loadings(u: np.ndarray, vh: np.ndarray, n_comp: int, mode: int) -> None:
    """
    消除 SVD ± 不定性：对每个主成分 k，将 u[:,k] 与 vh[k,:] 同乘 ±1（s 不变）。
    mode=1：列和规则（_pca_sign_should_flip_mode1）。
    mode=2：负载荷个数 > 正载荷个数则翻转；平局时回退为 mode=1。
    """
    if n_comp <= 0 or mode not in (1, 2):
        return
    k_max = int(min(n_comp, vh.shape[0], u.shape[1]))
    for k in range(k_max):
        lk = vh[k, :]
        flip = False
        if mode == 1:
            flip = _pca_sign_should_flip_mode1(lk)
        else:
            n_pos = int(np.sum(lk > 0.0))
            n_neg = int(np.sum(lk < 0.0))
            if n_neg > n_pos:
                flip = True
            elif n_neg == n_pos:
                flip = _pca_sign_should_flip_mode1(lk)
        if flip:
            u[:, k] *= -1.0
            vh[k, :] *= -1.0


def _tw_pca_from_corr_block(
    block: np.ndarray,
    cors_idx: List[int],
    anchor_idx: int,
    Tw: int,
    pca_components: int,
    pca_loading_sign_fix: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 block[:, cors_idx] 做列标准化 + SVD PCA，返回 (tw_pca Tw×K, ratio K, anchor_coef K)。
    其中 anchor_coef 是“当前股票（anchor_idx）在每个主成分向量中的系数”（若不在池内则为 0）。
    列数为 0 或方差近 0 时返回全零（与 build_samples 原逻辑一致）。
    """
    if len(cors_idx) == 0:
        return (
            np.zeros((Tw, pca_components), dtype=np.float32),
            np.zeros((pca_components,), dtype=np.float32),
            np.zeros((pca_components,), dtype=np.float32),
        )
    mat = block[:, cors_idx].astype(np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    scaler_w = StandardScaler()
    mat = scaler_w.fit_transform(mat)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, -PCA_INPUT_CLIP_ABS, PCA_INPUT_CLIP_ABS)
    n_comp = min(pca_components, mat.shape[1], mat.shape[0])
    if mat.shape[1] == 0 or np.nanvar(mat) <= 1e-12:
        return (
            np.zeros((Tw, pca_components), dtype=np.float32),
            np.zeros((pca_components,), dtype=np.float32),
            np.zeros((pca_components,), dtype=np.float32),
        )
    anchor_loc = -1
    try:
        anchor_loc = cors_idx.index(int(anchor_idx))
    except ValueError:
        anchor_loc = -1
    try:
        if pca_loading_sign_fix in (1, 2):
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            _align_pca_sign_from_loadings(u, vh, n_comp, int(pca_loading_sign_fix))
        else:
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
        tw_pca_raw = (u[:, :n_comp] * s[:n_comp]).astype(np.float32, copy=False)
        if anchor_loc >= 0 and anchor_loc < vh.shape[1]:
            coef_raw = vh[:n_comp, anchor_loc].astype(np.float32, copy=False)
        else:
            coef_raw = np.zeros((n_comp,), dtype=np.float32)
        if mat.shape[0] > 1:
            ev = (s * s) / float(mat.shape[0] - 1)
        else:
            ev = np.zeros_like(s, dtype=np.float32)
        total_ev = float(np.sum(ev))
        if total_ev > 0.0 and np.isfinite(total_ev):
            ratio_raw = (ev[:n_comp] / total_ev).astype(np.float32, copy=False)
        else:
            ratio_raw = np.zeros((n_comp,), dtype=np.float32)
    except Exception:
        tw_pca_raw = np.zeros((Tw, n_comp), dtype=np.float32)
        ratio_raw = np.zeros((n_comp,), dtype=np.float32)
        coef_raw = np.zeros((n_comp,), dtype=np.float32)
    if tw_pca_raw.shape[1] < pca_components:
        pad = np.zeros((Tw, pca_components - tw_pca_raw.shape[1]), dtype=np.float32)
        tw_pca = np.hstack([tw_pca_raw, pad])
    else:
        tw_pca = tw_pca_raw[:, :pca_components]
    ratio = np.asarray(ratio_raw, dtype=np.float32)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    if ratio.shape[0] < pca_components:
        pad = np.zeros((pca_components - ratio.shape[0],), dtype=np.float32)
        ratio = np.hstack([ratio, pad])
    else:
        ratio = ratio[:pca_components]
    coef = np.asarray(coef_raw, dtype=np.float32)
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    if coef.shape[0] < pca_components:
        pad = np.zeros((pca_components - coef.shape[0],), dtype=np.float32)
        coef = np.hstack([coef, pad])
    else:
        coef = coef[:pca_components]
    return tw_pca, ratio, coef


def build_samples(
    panel: pd.DataFrame,
    Ts: int,
    Tw: int,
    n_forward: int,
    corr_threshold: float = 0.5,
    corr_neg_threshold: float = 0.1,
    min_corr_stocks: int = 2,
    pca_components: int = 1,
    feature_names: Optional[List[str]] = None,
    pca_loading_sign_fix: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, float], np.ndarray]:
    """
    滑动窗口构建样本。
    统一构建“最大特征视图”样本：
    (Ts, d + 5*pca_components + 2)，其中额外部分依次为
    [正相关池 PCA, 正池解释方差占比, 正池本股系数, 负相关池 PCA, 负池解释方差占比, 正池股票数, 负池股票数]。
    正相关：corr>=corr_threshold；负相关：corr<=-corr_neg_threshold；两池各自独立做与原先相同的 PCA。
    若某池相关股票数 < min_corr_stocks，该池 PCA 相关特征置 0；
    fallback_mask（1）表示正池或负池至少有一侧触发上述 fallback。
    pca_loading_sign_fix：0 不定号；1 列和规则；2 正负个数规则，平局时同 1。
    返回: X (N, Ts, F), y_by_forward (N,5), valid_by_forward (N,5), meta。
    """
    pca_loading_sign_fix = int(pca_loading_sign_fix)
    if pca_loading_sign_fix not in (0, 1, 2):
        raise ValueError(f"pca_loading_sign_fix must be 0, 1, or 2, got {pca_loading_sign_fix}")
    # 约定：panel 由 load_panel 产出（已 trade_date=str 且按 trade_date, ts_code 排序）
    # 这里不再强制 sort/copy，避免在训练/调参时重复开销
    feature_names = feature_names or list(DEFAULT_TS_FEATURE_NAMES)
    panel["trade_date"] = panel["trade_date"].astype(str)
    dates = panel["trade_date"].unique().tolist()
    dates.sort()
    stocks = panel["ts_code"].unique().tolist()
    T, n_stocks = len(dates), len(stocks)
    d = len(feature_names)
    # base | pos_pca | pos_ratio | pos_coef | neg_pca | neg_ratio | n_pos | n_neg
    out_dim = d + 5 * pca_components + 2
    # 为保证与用/不用相关股 PCA 时样本集合完全一致：两种模式都从 Tw-1 对齐样本结束点。
    # 因此，无论 use_corr_pca 真假，都需要至少满足 [Tw-1, T-1) 的可用长度（至少支持 n_forward=1）。
    min_len = Tw + 1
    if Tw <= Ts:
        return (
            np.zeros((0, Ts, out_dim), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0, 5), dtype=np.uint8),
            [],
            {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "var": float("nan"), "n": 0.0},
            np.zeros((0,), dtype=np.uint8),
        )
    if T < min_len:
        return (
            np.zeros((0, Ts, out_dim), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0, 5), dtype=np.uint8),
            [],
            {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "var": float("nan"), "n": 0.0},
            np.zeros((0,), dtype=np.uint8),
        )

    # 1) 预构建：一次 pivot 出所有特征 + close，避免 6~7 次 pivot_table
    value_cols = feature_names + ["close"]
    wide = panel.pivot_table(index="trade_date", columns="ts_code", values=value_cols, aggfunc="first")
    wide = wide.reindex(index=dates)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([value_cols, stocks]))
    wide = wide.fillna(0.0)

    stock_features = np.empty((T, n_stocks, d), dtype=np.float32)
    for j, col in enumerate(feature_names):
        stock_features[:, :, j] = wide[col].to_numpy(dtype=np.float32, copy=False)
    close_arr = wide["close"].to_numpy(dtype=np.float32, copy=False)
    need_corr_info = "close_ret" in feature_names
    close_ret = stock_features[:, :, feature_names.index("close_ret")] if need_corr_info else None

    # 每只股票首次有有效收盘价的日期索引（上市或进入 panel 的首日）
    # 仅当 Ts/Tw 窗口完全落在 [first_idx, T) 内才构建样本，避免用 0 填充的上市前区间
    has_valid = (close_arr > 0).any(axis=0)
    first_idx = np.where(has_valid, np.argmax(close_arr > 0, axis=0).astype(np.intp), T)

    # 数据表未必从上市首日开始，用涨幅阈值判断异常日（如上市首日或超涨跌停）：|close_ret| > 0.11
    # 将异常日的收益率特征置零，避免标准化与相关矩阵被异常值主导
    if close_ret is not None:
        abnormal = np.abs(close_ret) > ABNORMAL_RET_THRESHOLD
        for nm in ["open_ret", "close_ret", "high_ret", "low_ret"]:
            if nm in feature_names:
                j = feature_names.index(nm)
                # abnormal: (T, n_stocks) 作用在前两维；对单列特征做布尔索引置零
                stock_features[:, :, j][abnormal] = 0.0

    list_X, list_y_all_forward, list_valid_all_forward, list_meta = [], [], [], []
    list_fallback_mask = []
    corr_counts = []
    # 统计 PCA 每个主成分的解释方差占比分布（把所有样本的所有主成分拉平后统计）
    pca_ratio_count = 0.0
    pca_ratio_sum = 0.0
    pca_ratio_sumsq = 0.0
    pca_ratio_min = float("inf")
    pca_ratio_max = float("-inf")
    i_start = Tw - 1
    for i in range(i_start, T - 1):
        # 相关矩阵只与时间窗口 (i, Tw) 有关，与 a_idx 无关；放到内层循环外避免重复计算
        if need_corr_info:
            block = close_ret[i - Tw + 1 : i + 1, :].astype(np.float32)  # type: ignore[union-attr]
            if block.shape[0] != Tw:
                continue
            # Guard against inf/huge values poisoning covariance/PCA.
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
            block = np.clip(block, -RET_BLOCK_CLIP_ABS, RET_BLOCK_CLIP_ABS)
            block_centered = block - np.nanmean(block, axis=0, keepdims=True)
            block_centered = np.nan_to_num(block_centered, nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = np.dot(block_centered.T, block_centered) / Tw
                std = np.sqrt(np.diag(cov))
                std[std <= 0] = 1.0
                corr_mat = cov / np.outer(std, std)
                corr_mat = np.clip(corr_mat, -1.0, 1.0)
                np.fill_diagonal(corr_mat, 1.0)
        for a_idx in range(n_stocks):
            # 当前日、未来日需有有效收盘价
            close_t = close_arr[i, a_idx]
            if close_t <= 0:
                continue
            # Ts 窗口 [i-Ts+1, i] 必须完全在该股票“首次有数据”之后，避免上市前无数据被填 0
            if (i - Ts + 1) < first_idx[a_idx]:
                continue
            # 为保证 plain / pca 在样本可用性上严格一致，两种模式都要求 Tw 窗口完整有效
            if (i - Tw + 1) < first_idx[a_idx]:
                continue
            xs = stock_features[i - Ts + 1 : i + 1, a_idx, :]
            if xs.shape[0] != Ts:
                continue
            y_by_forward = np.zeros((5,), dtype=np.float32)
            valid_by_forward = np.zeros((5,), dtype=np.uint8)
            for k in range(1, 6):
                i_fut_k = i + k
                if i_fut_k >= T:
                    continue
                close_fut_k = close_arr[i_fut_k, a_idx]
                if close_fut_k <= 0:
                    continue
                valid_by_forward[k - 1] = 1
                y_by_forward[k - 1] = close_fut_k / close_t - 1.0

            if need_corr_info:
                corr_a = corr_mat[a_idx, :]
                order = np.argsort(-corr_a)
                cors_idx_pos = [j for j in order if corr_a[j] >= corr_threshold]
                order_neg = np.argsort(corr_a)
                cors_idx_neg = [j for j in order_neg if corr_a[j] <= -corr_neg_threshold]
                corr_counts.append(float(len(cors_idx_pos) + len(cors_idx_neg)))
                fb_pos = len(cors_idx_pos) < min_corr_stocks
                fb_neg = len(cors_idx_neg) < min_corr_stocks
                is_fallback = int(fb_pos or fb_neg)
            else:
                cors_idx_pos = []
                cors_idx_neg = []
                is_fallback = 0
                fb_pos, fb_neg = False, False

            if need_corr_info:
                if fb_pos:
                    tw_pca_pos = np.zeros((Tw, pca_components), dtype=np.float32)
                    ratio_pos = np.zeros((pca_components,), dtype=np.float32)
                    coef_pos = np.zeros((pca_components,), dtype=np.float32)
                else:
                    tw_pca_pos, ratio_pos, coef_pos = _tw_pca_from_corr_block(
                        block, cors_idx_pos, a_idx, Tw, pca_components, pca_loading_sign_fix,
                    )
                if fb_neg:
                    tw_pca_neg = np.zeros((Tw, pca_components), dtype=np.float32)
                    ratio_neg = np.zeros((pca_components,), dtype=np.float32)
                else:
                    tw_pca_neg, ratio_neg, _ = _tw_pca_from_corr_block(
                        block, cors_idx_neg, a_idx, Tw, pca_components, pca_loading_sign_fix,
                    )
            else:
                tw_pca_pos = np.zeros((Tw, pca_components), dtype=np.float32)
                ratio_pos = np.zeros((pca_components,), dtype=np.float32)
                coef_pos = np.zeros((pca_components,), dtype=np.float32)
                tw_pca_neg = np.zeros((Tw, pca_components), dtype=np.float32)
                ratio_neg = np.zeros((pca_components,), dtype=np.float32)

            ts_pca_pos = tw_pca_pos[-Ts:, :]
            ts_pca_neg = tw_pca_neg[-Ts:, :]

            # 解释方差占比：正/负池各统计一遍
            for ratio in (ratio_pos, ratio_neg):
                if ratio.size:
                    pca_ratio_count += float(ratio.size)
                    pca_ratio_sum += float(ratio.sum())
                    pca_ratio_sumsq += float((ratio * ratio).sum())
                    pca_ratio_min = float(min(pca_ratio_min, ratio.min()))
                    pca_ratio_max = float(max(pca_ratio_max, ratio.max()))

            ts_ratio_pos = np.tile(ratio_pos[None, :], (Ts, 1)).astype(np.float32)
            ts_ratio_neg = np.tile(ratio_neg[None, :], (Ts, 1)).astype(np.float32)
            ts_coef_pos = np.tile(coef_pos[None, :], (Ts, 1)).astype(np.float32)
            ts_n_pos = np.full((Ts, 1), float(len(cors_idx_pos)), dtype=np.float32)
            ts_n_neg = np.full((Ts, 1), float(len(cors_idx_neg)), dtype=np.float32)
            x_full = np.hstack(
                [xs, ts_pca_pos, ts_ratio_pos, ts_coef_pos, ts_pca_neg, ts_ratio_neg, ts_n_pos, ts_n_neg]
            ).astype(np.float32)
            list_X.append(x_full)
            list_y_all_forward.append(y_by_forward)
            list_valid_all_forward.append(valid_by_forward)
            list_meta.append((stocks[a_idx], dates[i]))
            list_fallback_mask.append(is_fallback)

    if len(list_X) == 0:
        return (
            np.zeros((0, Ts, out_dim), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0, 5), dtype=np.uint8),
            [],
            {
                "mean": float("nan"), "min": float("nan"), "max": float("nan"), "var": float("nan"), "n": 0.0,
                "pca_ratio_mean": float("nan"), "pca_ratio_min": float("nan"), "pca_ratio_max": float("nan"),
                "pca_ratio_var": float("nan"), "pca_ratio_n": 0.0,
            },
            np.zeros((0,), dtype=np.uint8),
        )
    X = np.stack(list_X, axis=0)
    y_all_forward = np.asarray(list_y_all_forward, dtype=np.float32)
    valid_all_forward = np.asarray(list_valid_all_forward, dtype=np.uint8)
    if corr_counts:
        arr = np.asarray(corr_counts, dtype=np.float32)
        stats = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "var": float(arr.var(ddof=0)),
            "n": float(arr.size),
        }
    else:
        stats = {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "var": float("nan"), "n": 0.0}

    # 补充 PCA 解释方差占比统计信息（与 corr>=thr count stats 同级字段）
    if pca_ratio_count > 0:
        pca_ratio_mean = pca_ratio_sum / pca_ratio_count
        pca_ratio_var = pca_ratio_sumsq / pca_ratio_count - pca_ratio_mean * pca_ratio_mean
        if pca_ratio_var < 0:
            pca_ratio_var = 0.0  # 数值误差
        stats.update(
            {
                "pca_ratio_mean": float(pca_ratio_mean),
                "pca_ratio_min": float(pca_ratio_min),
                "pca_ratio_max": float(pca_ratio_max),
                "pca_ratio_var": float(pca_ratio_var),
                "pca_ratio_n": float(pca_ratio_count),
            }
        )
    else:
        stats.update(
            {
                "pca_ratio_mean": float("nan"),
                "pca_ratio_min": float("nan"),
                "pca_ratio_max": float("nan"),
                "pca_ratio_var": float("nan"),
                "pca_ratio_n": 0.0,
            }
        )
    fallback_mask = np.asarray(list_fallback_mask, dtype=np.uint8)
    return X, y_all_forward, valid_all_forward, list_meta, stats, fallback_mask


# --------------- 3. Dataset & 训练入口 ---------------

class StockTsDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        meta: Optional[List[Tuple[str, str]]] = None,
    ):
        self.y = y
        self.meta = meta
        if self.meta is not None and len(self.meta) != len(self.y):
            raise ValueError(f"StockTsDataset: len(meta)={len(self.meta)} != len(y)={len(self.y)}")
        if fit_scaler and X.size > 0:
            self.scaler = StandardScaler()
            N, T, F = X.shape
            self.scaler.fit(X.reshape(-1, F))
            self.X = self.scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
        else:
            self.scaler = scaler
            if scaler is not None and X.size > 0:
                N, T, F = X.shape
                self.X = scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
            else:
                self.X = X.astype(np.float32) if X.size > 0 else X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 若 NumPy 2.x 与 PyTorch 不兼容导致 torch.from_numpy 报错，则用 .tolist() 绕过
        try:
            x = torch.from_numpy(self.X[idx])
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                x = torch.tensor(self.X[idx].tolist(), dtype=torch.float32)
            else:
                raise
        y_t = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        if self.meta is not None:
            return x, y_t, self.meta[idx]
        return x, y_t


class TradeDateBatchSampler:
    """
    每个 batch 为同一 trade_date 的全部样本（截面），用于 ICLoss 与「日度截面相关」一致。
    __iter__ 每轮可打乱交易日顺序（shuffle_days=True）。
    与 DataLoader(..., batch_sampler=...) 配合使用；不可再传 batch_size / shuffle。
    """

    def __init__(
        self,
        meta: List[Tuple[str, str]],
        *,
        shuffle_days: bool,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        by_day: Dict[str, List[int]] = defaultdict(list)
        for i, m in enumerate(meta):
            by_day[str(m[1])].append(i)
        self._by_day = {d: sorted(idxs) for d, idxs in by_day.items()}
        self._day_keys = sorted(by_day.keys())
        self.shuffle_days = shuffle_days
        self.generator = generator

    def __len__(self) -> int:
        return len(self._day_keys)

    def __iter__(self) -> Iterator[List[int]]:
        keys = list(self._day_keys)
        if self.shuffle_days and keys:
            n = len(keys)
            if self.generator is not None:
                perm = torch.randperm(n, generator=self.generator).tolist()
                keys = [keys[i] for i in perm]
            else:
                random.shuffle(keys)
        for d in keys:
            yield self._by_day[d]


def train_stock_ts(
    data_source: Literal["hs300", "sp500"],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    Ts: int,
    Tw: int,
    n_forward: int,
    corr_threshold: float,
    corr_neg_threshold: float,
    min_corr_stocks: int,
    corr_fallback_mode: Literal["zero", "drop"],
    pca_components: int,
    pca_loading_sign_fix: int,
    with_pca_ratio: bool,
    with_corr_count_feature: bool,
    with_pca_self_coef_feature: bool,
    model_type: str,
    batch_size: int,
    epochs: int,
    lr: float,
    hidden_size: int,
    pca_lstm_hidden_size: Optional[int],
    num_layers: int,
    dropout: float,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    seed: int,
    device: Optional[torch.device],
    use_cache: bool,
    use_corr_pca: bool,
    pos_sse: bool,
    neg_sse: bool,
    features: Optional[List[str]],
    split_mode: Literal["random", "time"],
    top_k: int,
    topk_pred_threshold: float,
    backtest_cost_preset: Literal["none", "auto", "cn", "us", "hk", "uk", "jp"],
    rebalance_emax: Optional[int],
    initial_cash: float,
    annual_trading_days: int,
    lite_info: bool,
    training_loss: Literal["mse", "mse+", "ic", "mon", "bce"],
    bce_top_k: int,
    bce_bottom_k: int,
    model_extra_kwargs: Optional[Dict[str, Any]],
    daily_rank_ic_dir: Optional[str] = None,
) -> nn.Module:
    """
    训练流程：仅在 [train_start, train_end] 上 load_panel + build_samples 一次。
    要求 train_start <= test_start <= test_end <= train_end（YYYYMMDD 字符串序）。
    - trade_date < test_start 的样本为「开发池」，按 80% / 20% 划分为 train / val，早停看 val；
    - split_mode="random" 时在开发池内随机划分；split_mode="time" 时按 (trade_date, ts_code) 严格时间顺序划分。
    - trade_date 在 [test_start, test_end] 内的样本为独立测试集（从同一份 X,y,meta 中切片，不再单独构建测试 panel）。
    - trade_date > test_end 的样本既不参与开发池划分也不进入独立测试集（若存在会打印提示）。
    - training_loss：
      * mse：MSELoss + 常规固定 batch_size 小批量（mci-gru 例外：强制按 trade_date 截面 batch，等价日度 MSE 平均）；
      * mse+：MSELoss + 每个 batch 为同一 trade_date 的截面样本；
      * ic：ICLoss + 每个 batch 为同一 trade_date 的截面样本（在该 batch 内算 Pearson 相关并取负）。
      * mon：DeltaLag 论文式 monotonic logistic ranking loss（按日截面、每个无序对 i<j 一项）。
      * bce：BCEWithLogitsLoss + 论文式「按日 top/bottom-k」二分类标签（中间样本不参与训练）。
    - model_type=mci-gru：官方 WinstonLiyt/MCI-GRU 整网（AttentionGRU+GAT+CrossAttn+SelfAttn+GAT），需 torch_geometric；
      每个 batch 为同一交易日的全截面；构图用 close_ret 相关矩阵（lookback=judge 见 model_extra_kwargs：mci_corr_lookback、mci_judge_value）。
    use_corr_pca=True 时，样本构建阶段始终保留样本并在相关股票数 < min_corr_stocks 时将 PCA 相关特征置 0；
    正相关池用 corr_threshold，负相关池用 corr_neg_threshold（纳入 corr<=-corr_neg_threshold）；
    若 corr_fallback_mode='drop'，则在训练/验证/测试集合切分前过滤掉这些 fallback 样本。
    use_corr_pca=False 时仅用 (Ts, d) 做单任务时序预测，不做 Tw/相关股/PCA。
    pos_sse / neg_sse：是否把正相关池 / 负相关池对应的列拼入模型输入（PCA、方差占比、池股票数）；默认正池开、负池关。
    二者不参与样本缓存键；缓存始终为完整「最大特征视图」，仅在装入内存后从统一布局中按需拼接列（关闭的一侧不拼接，而非置零）。
    with_pca_ratio / with_corr_count_feature / with_pca_self_coef_feature 默认与 main.py 一致为 False
    （主流：仅拼接两侧 PCA 序列，不拼解释方差、池计数、本股系数）。
    pca_loading_sign_fix：0 不定号；1 载荷列和规则；2 载荷正负个数规则，平局同 1。
    回测：backtest_cost_preset / rebalance_emax 传入 _backtest_testset_cr_ar_sr_compare（见该函数说明）。
    - daily_rank_ic_dir：若设置，在其下的 t_test 子目录写出测试集「每日截面 Rank IC」CSV，
      文件名为 daily_rank_ic_{model_type}_{data_source}_{test_start}_{test_end}.csv。
    """
    pca_loading_sign_fix = int(pca_loading_sign_fix)
    if pca_loading_sign_fix not in (0, 1, 2):
        raise ValueError(f"pca_loading_sign_fix must be 0, 1, or 2, got {pca_loading_sign_fix}")
    if seed is not None:
        set_seed(seed)
    if _is_spatialsyn_family(model_type):
        if not use_corr_pca:
            print(
                "[SpatialSyn family] use_corr_pca=False is incompatible (needs correlated-stock PCA "
                "columns); forcing use_corr_pca=True."
            )
        use_corr_pca = True
    if device is not None:
        pass
    else:
        # Prefer CUDA; then Apple Silicon (MPS) when available; finally CPU.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # If user explicitly requested a device but it's unavailable, fall back gracefully.
    if getattr(device, "type", None) == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but not available; falling back to cpu")
        device = torch.device("cpu")
    if getattr(device, "type", None) == "mps":
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_available:
            print("[device] MPS requested but not available; falling back to cpu")
            device = torch.device("cpu")
    # Always print selected device once so you can confirm MPS/CUDA/CPU is really used (hidden in lite_info).
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print("")
    if not lite_info:
        
        print(
            f"[device] selected={device} | cuda_available={torch.cuda.is_available()} | mps_available={mps_available}"
        )
    profile = os.environ.get("PROFILE_TS", "0") == "1"
    timings: Dict[str, float] = {}

    def _sync_if_needed() -> None:
        # Make timings comparable when running on async backends (MPS/CUDA).
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and getattr(torch.backends, "mps", None) is not None:
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    def _time_section(name: str, start: float) -> None:
        _sync_if_needed()
        timings[name] = time.perf_counter() - start

    do_print = not lite_info
    if do_print and _is_spatialsyn_family(model_type):
        _mtu_sp = str(model_type).upper()
        if _mtu_sp in ("SPATIALSYN_PRO", "SPATIALSYN_PRO_LN"):
            _branches = "stock branch LSTM + PCA-component branch LSTM, then added before MLP"
        else:
            _branches = "stock branch LSTM + PCA-component branch LSTM"
        print(
            f"{model_type}: use_corr_pca=True (same sample layout as LSTM + correlated-stock PCA); "
            f"{_branches}, fused by MLP."
        )
    if do_print and (not use_corr_pca):
        print("Mode: plain single-task (Ts, d) only — no Tw / correlated stocks / PCA.")
    features = features or list(DEFAULT_TS_FEATURE_NAMES)
    allowed = set(DEFAULT_TS_FEATURE_NAMES)
    unknown = [f for f in features if f not in allowed]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Allowed: {sorted(allowed)}")
    if "close_ret" not in features:
        raise ValueError("统一 samples 缓存构建需要 features 包含 close_ret（用于相关系数与fallback判定）")
    if _is_spatialsyn_family(model_type) and pca_components < 1:
        raise ValueError("SpatialSyn / SpatialSyn_Pro requires pca_components >= 1")

    if not (train_start <= test_start <= test_end <= train_end):
        raise ValueError(
            f"日期须满足 train_start <= test_start <= test_end <= train_end，"
            f"当前: {train_start=}, {test_start=}, {test_end=}, {train_end=}"
        )
    if min_corr_stocks < 0:
        raise ValueError(f"min_corr_stocks must be >= 0, got {min_corr_stocks}")
    if corr_fallback_mode not in {"zero", "drop"}:
        raise ValueError(f"corr_fallback_mode must be 'zero' or 'drop', got {corr_fallback_mode}")

    panel_t0 = time.perf_counter() if profile else 0.0
    key_panel = _panel_cache_key(data_source, train_start, train_end)
    panel_full = _load_cached_panel(key_panel) if use_cache else None

    if panel_full is not None:
        if do_print:
            print(f"Panel from cache | {train_start}~{train_end}")
    else:
        if do_print:
            print(f"Loading panel | {train_start}~{train_end}...")
        panel_full = load_panel(train_start, train_end, data_source)
        if use_cache:
            _save_cached_panel(key_panel, panel_full)
            if do_print:
                print(f"Panel cached: {key_panel}.pkl")
    if profile:
        _time_section("panel_load", panel_t0)

    n_stocks = panel_full["ts_code"].nunique()
    n_dates = panel_full["trade_date"].nunique()
    if do_print:
        print(
            f"Panel: {len(panel_full)} rows | {n_stocks} stocks × {n_dates} dates | {train_start}~{train_end}"
        )

    samples_t0 = time.perf_counter() if profile else 0.0
    key_samples = _samples_cache_key(
        data_source=data_source, start=train_start, end=train_end,
        Ts=Ts, Tw=Tw, n_forward=n_forward, corr_threshold=corr_threshold,
        corr_neg_threshold=corr_neg_threshold,
        min_corr_stocks=min_corr_stocks, pca_components=pca_components, features=features,
        pca_loading_sign_fix=pca_loading_sign_fix,
    )

    need_fallback_mask = corr_fallback_mode == "drop"
    cached = _load_cached_samples(key_samples, require_fallback_mask=need_fallback_mask) if use_cache else None
    if cached is not None:
        X_all, y_all_forward, valid_all_forward, meta_all, stats_all, fallback_mask_all = cached
        if do_print:
            print(f"Loaded samples from cache | {key_samples}.npz")
        if stats_all:
            if do_print:
                print(
                    f"corr>=thr count stats | mean={stats_all.get('mean'):.2f} "
                    f"min={stats_all.get('min'):.0f} max={stats_all.get('max'):.0f} "
                    f"var={stats_all.get('var'):.2f} n={int(stats_all.get('n', 0))}"
                    f" | pca explained_var_ratio stats | mean={stats_all.get('pca_ratio_mean', float('nan')):.4f} "
                    f"min={stats_all.get('pca_ratio_min', float('nan')):.4f} "
                    f"max={stats_all.get('pca_ratio_max', float('nan')):.4f} "
                    f"var={stats_all.get('pca_ratio_var', float('nan')):.6f} "
                    f"n={int(stats_all.get('pca_ratio_n', 0))}"
                )
    else:
        if do_print:
            print("Building samples...")
        X_all, y_all_forward, valid_all_forward, meta_all, stats_all, fallback_mask_all = build_samples(
            panel_full, Ts=Ts, Tw=Tw, n_forward=n_forward, corr_threshold=corr_threshold,
            corr_neg_threshold=corr_neg_threshold,
            min_corr_stocks=min_corr_stocks, pca_components=pca_components, feature_names=features,
            pca_loading_sign_fix=pca_loading_sign_fix,
        )
        
        _save_cached_samples(
                key_samples,
                X_all,
                y_all_forward,
                valid_all_forward,
                meta_all,
                fallback_mask=fallback_mask_all,
                stats=stats_all,
            )
        if do_print:
                print(f"Samples cached: {key_samples}.npz")    

        if stats_all and stats_all.get("n", 0) > 0:
            if do_print:
                print(
                    f"corr>=thr count stats | mean={stats_all.get('mean'):.2f} "
                    f"min={stats_all.get('min'):.0f} max={stats_all.get('max'):.0f} "
                    f"var={stats_all.get('var'):.2f} n={int(stats_all.get('n', 0))}"
                    f" | pca explained_var_ratio stats | mean={stats_all.get('pca_ratio_mean', float('nan')):.4f} "
                    f"min={stats_all.get('pca_ratio_min', float('nan')):.4f} "
                    f"max={stats_all.get('pca_ratio_max', float('nan')):.4f} "
                    f"var={stats_all.get('pca_ratio_var', float('nan')):.6f} "
                    f"n={int(stats_all.get('pca_ratio_n', 0))}"
                )
    if profile:
        _time_section("samples_load_or_build", samples_t0)

    forward_idx = n_forward - 1
    keep_forward = valid_all_forward[:, forward_idx] == 1
    X_all = X_all[keep_forward]
    y_all = y_all_forward[keep_forward, forward_idx]
    meta_all = [meta_all[i] for i, k in enumerate(keep_forward) if k]
    fallback_mask_all = fallback_mask_all[keep_forward]

    if len(y_all) == 0:
        raise RuntimeError("No samples in [train_start, train_end]. Check date range and data_source.")

    if corr_fallback_mode == "drop":
        keep_mask = fallback_mask_all == 0
        dropped = int((~keep_mask).sum())
        if do_print:
            print(f"corr_fallback_mode=drop: filtered fallback samples {dropped}/{len(y_all)}")
        X_all = X_all[keep_mask]
        y_all = y_all[keep_mask]
        meta_all = [meta_all[i] for i, k in enumerate(keep_mask) if k]
        fallback_mask_all = fallback_mask_all[keep_mask]

    # Unified cached X layout（只读切片，不修改缓存内容）:
    # [base_d | pca_pos(pc) | ratio_pos(pc) | coef_pos(pc) | pca_neg(pc) | ratio_neg(pc) | n_pos(1) | n_neg(1)].
    # 主流：三个 with_* 皆 False 时，仅拼接正/负池 PCA 段，F = d_base + pc * (pos_sse + neg_sse)。
    d_base = len(features)
    pc = pca_components
    c_pos_pca = d_base
    c_pos_ratio = d_base + pc
    c_pos_coef = d_base + 2 * pc
    c_neg_pca = d_base + 3 * pc
    c_neg_ratio = d_base + 4 * pc
    c_npos = d_base + 5 * pc
    c_nneg = d_base + 5 * pc + 1
    if use_corr_pca:
        if _is_spatialsyn_family(model_type) and (not pos_sse) and (not neg_sse):
            raise ValueError(
                "SpatialSyn / SpatialSyn_Pro 需要至少一侧相关池特征：请打开 pos_sse 或 neg_sse（或二者）。"
            )
        parts: List[np.ndarray] = []
        if pos_sse:
            parts.append(X_all[:, :, c_pos_pca : c_pos_pca + pc])
        if neg_sse:
            parts.append(X_all[:, :, c_neg_pca : c_neg_pca + pc])
        if with_pca_ratio:
            if pos_sse:
                parts.append(X_all[:, :, c_pos_ratio : c_pos_ratio + pc])
            if neg_sse:
                parts.append(X_all[:, :, c_neg_ratio : c_neg_ratio + pc])
        if with_pca_self_coef_feature:
            if pos_sse:
                parts.append(X_all[:, :, c_pos_coef : c_pos_coef + pc])
        if with_corr_count_feature:
            corr_chunks: List[np.ndarray] = []
            if pos_sse:
                corr_chunks.append(X_all[:, :, c_npos : c_npos + 1])
            if neg_sse:
                corr_chunks.append(X_all[:, :, c_nneg : c_nneg + 1])
            if corr_chunks:
                parts.append(
                    corr_chunks[0]
                    if len(corr_chunks) == 1
                    else np.concatenate(corr_chunks, axis=2)
                )
        if parts:
            X_all = np.concatenate([X_all[:, :, :d_base], *parts], axis=2)
        else:
            X_all = X_all[:, :, :d_base]
    else:
        X_all = X_all[:, :, :d_base]

    mci_graph = None
    if _is_mci_gru(model_type):
        mx = model_extra_kwargs or {}
        mci_j = float(mx.get("mci_judge_value", 0.8))
        mci_lb = int(mx.get("mci_corr_lookback", 250))
        tickers_union = sorted({m[0] for m in meta_all})
        ext_start = _shift_ymd(train_start, -1200)
        if do_print:
            print(
                f"[MCI-GRU] 官方 StockPredictionModel（AttentionGRU+GAT+CrossAttn+SelfAttn+GAT）；"
                f"每个 train/val batch = 同一 trade_date 截面；"
                f"构图 panel {ext_start}~{train_end}，corr lookback={mci_lb}，edge if corr>{mci_j}"
            )
        panel_mci = load_panel(ext_start, train_end, data_source)
        mci_graph = build_mci_gru_graph_builder_from_panel(
            panel_mci,
            tickers=tickers_union,
            lookback=mci_lb,
            judge=mci_j,
            return_col="close_ret",
        )

    split_t0 = time.perf_counter() if profile else 0.0
    td = np.array([m[1] for m in meta_all], dtype=str)
    mask_indep = (td >= test_start) & (td <= test_end)
    mask_dev = td < test_start
    n_after = int(np.sum(td > test_end))
    if n_after:
        if do_print:
            print(
                f"Note: {n_after} samples with trade_date > test_end (within panel) are excluded from "
                "train/val and 独立测试集."
            )

    idx_test = np.where(mask_indep)[0]
    idx_dev = np.where(mask_dev)[0]
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]
    meta_test = [meta_all[i] for i in idx_test]
    X_pool = X_all[idx_dev]
    y_pool = y_all[idx_dev]
    meta_pool = [meta_all[i] for i in idx_dev]

    if do_print:
        print(
            f"样本划分: 全量={len(y_all)} | 开发池(trade_date<{test_start})={len(y_pool)} | "
            f"测试[{test_start},{test_end}]={len(y_test)}"
        )
    if len(y_pool) == 0:
        raise RuntimeError(
            f"No development-pool samples (trade_date < {test_start}). "
            "Widen [train_start, train_end] or choose a later test_start."
        )
    if len(y_test) == 0:
        if do_print:
            print(
                f"Warning: 测试集样本数为 0（区间 [{test_start},{test_end}] 内无有效样本）。"
            )

    # 开发池按 80% / 20% 划分 train / val；Test 使用 [test_start, test_end] 区间样本
    idx = np.arange(len(y_pool))
    if split_mode == "random":
        rs = seed if seed is not None else 42
        i_tr, i_val = train_test_split(idx, test_size=0.2, random_state=rs)
    elif split_mode == "time":
        # 严格按日期边界切分：同一天样本不能分到多个集合（仅 train/val）
        # 先按 (trade_date, ts_code) 排序，保证可复现
        idx_sorted = np.array(sorted(idx, key=lambda j: (meta_pool[j][1], meta_pool[j][0])), dtype=int)
        pool_dates = np.array([meta_pool[j][1] for j in idx_sorted], dtype=str)
        unique_dates = np.array(sorted(np.unique(pool_dates)), dtype=str)
        n_dates = len(unique_dates)

        if n_dates == 1:
            # 仅一天时无法做时序二划分，退化为全量 train
            tr_dates = set(unique_dates.tolist())
            val_dates = set()
        elif n_dates == 2:
            tr_dates = {unique_dates[0]}
            val_dates = {unique_dates[1]}
        else:
            n_tr_d = int(n_dates * 0.8)
            n_val_d = n_dates - n_tr_d
            if n_tr_d <= 0:
                n_tr_d = 1
                n_val_d = n_dates - n_tr_d
            if n_val_d <= 0:
                n_val_d = 1
                n_tr_d = n_dates - n_val_d

            tr_dates = set(unique_dates[:n_tr_d].tolist())
            val_dates = set(unique_dates[n_tr_d:n_tr_d + n_val_d].tolist())

        i_tr = np.array([j for j in idx_sorted if meta_pool[j][1] in tr_dates], dtype=int)
        i_val = np.array([j for j in idx_sorted if meta_pool[j][1] in val_dates], dtype=int)
    else:
        raise ValueError(f"Unknown split_mode={split_mode}, expected 'random' or 'time'")

    X_tr = X_pool[i_tr]
    y_tr = y_pool[i_tr]
    meta_tr = [meta_pool[j] for j in i_tr]
    X_val = X_pool[i_val]
    y_val = y_pool[i_val]
    meta_val = [meta_pool[j] for j in i_val]
    X_te = X_test
    y_te = y_test
    meta_te = meta_test
    y_tr_target = y_tr
    y_val_target = y_val
    if _is_mci_gru(model_type):
        # Official MCI-GRU uses per-day rank_labeling as supervision.
        y_tr_target = _daily_rank_labeling(y_tr, meta_tr)
        y_val_target = _daily_rank_labeling(y_val, meta_val)
        if do_print:
            print("[MCI-GRU] training labels use daily rank_labeling (pct rank).")
    if training_loss == "bce":
        mt_upper = str(model_type).upper()
        if mt_upper == "DTML":
            # Keep DTML consistent with the paper's movement-label definition.
            y_tr_bce = _binary_labels_by_return_sign(y_tr)
            y_val_bce = _binary_labels_by_return_sign(y_val)
        else:
            # THGNN and other BCE users: daily top/bottom-k labels.
            y_tr_bce = _daily_top_bottom_binary_labels(
                y_tr, meta_tr, top_k=int(bce_top_k), bottom_k=int(bce_bottom_k)
            )
            y_val_bce = _daily_top_bottom_binary_labels(
                y_val, meta_val, top_k=int(bce_top_k), bottom_k=int(bce_bottom_k)
            )
        tr_mask = np.isfinite(y_tr_bce)
        val_mask = np.isfinite(y_val_bce)
        X_tr, y_tr_target = X_tr[tr_mask], y_tr_bce[tr_mask]
        X_val, y_val_target = X_val[val_mask], y_val_bce[val_mask]
        meta_tr = [m for m, keep in zip(meta_tr, tr_mask) if keep]
        meta_val = [m for m, keep in zip(meta_val, val_mask) if keep]
        if len(y_tr_target) == 0 or len(y_val_target) == 0:
            raise RuntimeError(
                "BCE labeling produced empty train/val set; "
                "please reduce bce_top_k/bce_bottom_k or expand data range."
            )

    _mci_meta = _is_mci_gru(model_type)
    ds_train = StockTsDataset(
        X_tr, y_tr_target, scaler=None, fit_scaler=True, meta=meta_tr if _mci_meta else None
    )
    scaler = ds_train.scaler
    ds_val = StockTsDataset(
        X_val, y_val_target, scaler=scaler, fit_scaler=False, meta=meta_val if _mci_meta else None
    )
    ds_test = StockTsDataset(X_te, y_te, scaler=scaler, fit_scaler=False, meta=meta_te if _mci_meta else None)

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    use_day_batch = training_loss in ("ic", "mse+", "mon", "bce") or _is_mci_gru(model_type)
    _mci_collate = _mci_gru_collate if _is_mci_gru(model_type) else None
    if use_day_batch:
        bs_train = TradeDateBatchSampler(meta_tr, shuffle_days=True, generator=g)
        bs_val = TradeDateBatchSampler(meta_val, shuffle_days=False, generator=None)
        dl_train = DataLoader(ds_train, batch_sampler=bs_train, collate_fn=_mci_collate)
        dl_val = DataLoader(ds_val, batch_sampler=bs_val, collate_fn=_mci_collate)
        if do_print and not lite_info:
            if _is_mci_gru(model_type):
                mode_label = "MCI-GRU（官方整网 + 日截面 batch）"
            elif training_loss == "ic":
                mode_label = "ICLoss"
            elif training_loss == "mon":
                mode_label = "MonotonicRankingLoss"
            elif training_loss == "bce":
                mode_label = "BCE/day-topbottom"
            else:
                mode_label = "MSE/day-batch"
            print(
                f"[{mode_label}] 每个 train/val batch = 一个 trade_date 的全部样本 "
                f"（train 共 {len(bs_train)} 个交易日；val 共 {len(bs_val)} 个）；"
                f"忽略 batch_size={batch_size}"
            )
    else:
        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=g,
            collate_fn=_mci_collate,
        )
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=_mci_collate)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=_mci_collate)
    if profile:
        _time_section("split_scale_dataloaders", split_t0)

    input_dim = X_all.shape[2]
    _pca_lstm_h = pca_lstm_hidden_size if pca_lstm_hidden_size is not None else hidden_size
    _layout_base = len(features)
    _n_pool_side = int(pos_sse) + int(neg_sse)
    _ratio_dim = (pca_components * _n_pool_side) if (use_corr_pca and with_pca_ratio) else 0
    _coef_dim = (pca_components if (use_corr_pca and with_pca_self_coef_feature and pos_sse) else 0)
    _corr_dim = _n_pool_side if (use_corr_pca and with_corr_count_feature) else 0
    if _is_spatialsyn_family(model_type) and use_corr_pca:
        _exp_in = _layout_base + pca_components * _n_pool_side + _ratio_dim + _coef_dim + _corr_dim
        if _exp_in != input_dim:
            raise RuntimeError(
                "SpatialSyn / SpatialSyn_Pro 输入维与拼接逻辑不一致（请检查 pos_sse/neg_sse 与 with_pca_ratio/with_pca_self_coef_feature/with_corr_count_feature）。"
                f" 期望 base+pca+ratio+coef+corr={_exp_in}（base={_layout_base}, pca={pca_components * _n_pool_side}, "
                f"ratio={_ratio_dim}, coef={_coef_dim}, corr={_corr_dim}），实际张量 F={input_dim}。"
                f" pos_sse={pos_sse}, neg_sse={neg_sse}, with_pca_ratio={with_pca_ratio}, "
                f"with_pca_self_coef_feature={with_pca_self_coef_feature}, with_corr_count_feature={with_corr_count_feature}"
            )

    # 训练开始前打印关键信息（lite_info=只保留最基本的对比行）
    if lite_info:
        print(
            f"data_source={data_source}, model={model_type}, use_corr_pca={use_corr_pca}, "
            f"pos_sse={pos_sse}, neg_sse={neg_sse}, "
            f"corr_threshold={corr_threshold}, corr_neg_threshold={corr_neg_threshold}, "
            f"pca_loading_sign_fix={pca_loading_sign_fix}, "
            f"with_pca_ratio={with_pca_ratio}, with_pca_self_coef_feature={with_pca_self_coef_feature}, "
            f"with_corr_count_feature={with_corr_count_feature}, "
            f"training_loss={training_loss}"
        )
    else:
        # 便于自动记录：打印主要超参数与数据切分规模
        print("--- Hyperparams ---")
        print(
            f"data_source={data_source} panel={train_start}~{train_end} test={test_start}~{test_end} "
            f"Ts={Ts} Tw={Tw} n_forward={n_forward} corr_threshold={corr_threshold} "
            f"corr_neg_threshold={corr_neg_threshold} min_corr_stocks={min_corr_stocks} "
            f"corr_fallback_mode={corr_fallback_mode} use_corr_pca={use_corr_pca} "
            f"pos_sse={pos_sse} neg_sse={neg_sse}"
        )
        if use_corr_pca:
            print(f"pca_components={pca_components}")
            print(f"pca_loading_sign_fix={pca_loading_sign_fix}")
            print(f"with_pca_ratio={with_pca_ratio}")
            print(f"with_pca_self_coef_feature={with_pca_self_coef_feature}")
            print(f"with_corr_count_feature={with_corr_count_feature}")
        print(f"features={features}")

    model_extra_kwargs = dict(model_extra_kwargs or {})
    _raw_seq_len = model_extra_kwargs.get("seq_len")
    _seq_len = Ts if _raw_seq_len is None else int(_raw_seq_len)
    if not lite_info:
        _mtu = model_type.upper()
        _tf_h = ""
        if _mtu in ("INFORMER", "AUTOFORMER"):
            _d_model = model_extra_kwargs.get("d_model")
            _n_heads = model_extra_kwargs.get("n_heads")
            _e_layers = model_extra_kwargs.get("e_layers")
            _d_layers = model_extra_kwargs.get("d_layers")
            _d_ff = model_extra_kwargs.get("d_ff")
            _factor = model_extra_kwargs.get("factor")
            _distil = model_extra_kwargs.get("distil")
            _moving_avg = model_extra_kwargs.get("moving_avg")
            _tf_h = (
                f" d_model={_d_model} n_heads={_n_heads} enc_layers={_e_layers} dec_layers={_d_layers} "
                f"d_ff={_d_ff} factor={_factor} seq_len={_seq_len}"
            )
            if _mtu == "INFORMER":
                _tf_h += f" distil={_distil}"
            else:
                _tf_h += f" moving_avg={_moving_avg}"
        elif _mtu in ("SPATIALSYN", "SPATIALSYN_PRO", "SPATIALSYN_LN", "SPATIALSYN_PRO_LN"):
            _tf_h = f" pca_lstm_hidden={_pca_lstm_h}"
        print(
            f"model={model_type} batch={batch_size} epochs={epochs} lr={lr} "
            f"hidden={hidden_size} layers={num_layers} dropout={dropout}{_tf_h} "
            f"training_loss={training_loss} "
            f"early_stop={early_stopping} patience={patience} min_delta={min_delta} seed={seed} split_mode={split_mode}"
        )
        print(
            f"input_dim={input_dim} | train/val/test samples: {len(ds_train)}/{len(ds_val)}/{len(y_te)}"
        )
        print("-------------------")

    model_t0 = time.perf_counter() if profile else 0.0
    model_kwargs: Dict[str, Any] = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": _seq_len,
    }
    if model_extra_kwargs:
        # Allow model-specific kwargs from CLI local overrides.
        # Keep seq_len fallback behavior stable: None means "use Ts" instead of
        # overriding to an invalid None for wrappers that require integer seq_len.
        _extra = dict(model_extra_kwargs)
        if _extra.get("seq_len", "__MISSING__") is None:
            _extra.pop("seq_len", None)
        model_kwargs.update(_extra)
    if _is_spatialsyn_family(model_type):
        model_kwargs.update(
            base_dim=_layout_base,
            pca_dim=pca_components * _n_pool_side,
            ratio_dim=_ratio_dim,
            coef_dim=_coef_dim,
            corr_dim=_corr_dim,
            pca_lstm_hidden_size=_pca_lstm_h,
        )
    if str(model_type).upper() in ("MAMBA",):
        # Fused mamba-ssm is CUDA-only; pick MambaTorch when training on MPS/CPU.
        model_kwargs = {**model_kwargs, "mamba_target_device": device}
    for _k in ("mci_judge_value", "mci_corr_lookback"):
        model_kwargs.pop(_k, None)
    # DeltaLag-only hyperparams may appear in model_extra from CLI; strip for other
    # models so future wrappers without ``**kwargs`` are not broken.
    if str(model_type).upper() != "DELTALAG":
        for _k in list(model_kwargs.keys()):
            if str(_k).startswith("deltalag_"):
                model_kwargs.pop(_k, None)
    model = build_model(model_type, input_dim, "reg", **model_kwargs)
    model.to(device)
    # Confirm the device where model parameters actually live.
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        first_param_device = None
    if not lite_info:
        print(f"[model] first_param_device={first_param_device}")
    loss_fn = _make_training_loss(training_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if profile:
        _time_section("model_build_to_device", model_t0)

    best_val_loss = None
    best_epoch = 0
    best_sd = None
    epochs_no_improve = 0

    train_t0 = time.perf_counter() if profile else 0.0
    per_batch_mean = use_day_batch
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr_batches = 0
        for batch in dl_train:
            optimizer.zero_grad()
            out, yb = _forward_stock_ts_batch(
                model, batch, device, mci_graph=mci_graph
            )
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if per_batch_mean:
                tr_loss += loss.item()
                n_tr_batches += 1
            else:
                tr_loss += loss.item() * batch[0].shape[0]
        if per_batch_mean:
            tr_loss /= max(n_tr_batches, 1)
        else:
            tr_loss /= len(ds_train)

        val_loss = _eval_loss_stock_ts(
            model,
            dl_val,
            loss_fn,
            device,
            per_batch_mean=per_batch_mean,
            mci_graph=mci_graph,
        )
        improved = best_val_loss is None or val_loss < best_val_loss - min_delta
        if do_print:
            print(
                f"Epoch {ep:>2}/{epochs} | TrainLoss {tr_loss:.5f} | ValLoss {val_loss:.5f}",
                end="",
            )
        if early_stopping:
            if improved:
                best_val_loss = val_loss
                best_epoch = ep
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                if do_print:
                    print(" *")
            else:
                epochs_no_improve += 1
                if do_print:
                    print(f" ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    if do_print or lite_info:
                        print(
                            f"Early stopping at epoch {ep}. Best epoch: {best_epoch}"
                        )
                    break
        else:
            if do_print:
                print()
    if profile:
        _time_section("train_loop", train_t0)

    if early_stopping and best_sd is not None:
        model.load_state_dict(best_sd)
        model.to(device)
        if do_print:
            print(f"Restored best model from epoch {best_epoch}")

    # 汇报：用「预测+标签同序」接口保证 pred[i] 与 y_true[i] 严格对应同一样本，避免错位
    model.eval()
    if use_day_batch:
        dl_train_eval = DataLoader(
            ds_train,
            batch_sampler=TradeDateBatchSampler(meta_tr, shuffle_days=False, generator=None),
            collate_fn=_mci_collate,
        )
    else:
        dl_train_eval = DataLoader(
            ds_train, batch_size=batch_size, shuffle=False, collate_fn=_mci_collate
        )
    eval_t0 = time.perf_counter() if profile else 0.0
    with torch.no_grad():
        train_pred, y_tr_aligned = _predict_stock_ts_with_labels(
            model, dl_train_eval, device, mci_graph=mci_graph
        )
        val_pred, y_val_aligned = _predict_stock_ts_with_labels(
            model, dl_val, device, mci_graph=mci_graph
        )
        test_pred, y_te_aligned = _predict_stock_ts_with_labels(
            model, dl_test, device, mci_graph=mci_graph
        )
    if profile:
        _time_section("predict_eval", eval_t0)
    # 长度校验，防止 DataLoader 与原始 y 长度不一致导致错位
    assert len(y_tr_aligned) == len(train_pred), "Train pred 与 y 长度不一致"
    assert len(y_val_aligned) == len(val_pred), "Val pred 与 y 长度不一致"
    assert len(y_te_aligned) == len(test_pred), "Test pred 与 y 长度不一致"
    _daily_ric_path: Optional[str] = None
    if daily_rank_ic_dir:
        _ttest_dir = os.path.join(daily_rank_ic_dir, "t_test")
        os.makedirs(_ttest_dir, exist_ok=True)
        _safe_m = re.sub(r"[^a-zA-Z0-9._-]+", "_", (model_type or "model").strip() or "model")
        _safe_ds = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(data_source).strip() or "dataset")
        _daily_ric_path = os.path.join(
            _ttest_dir,
            f"daily_rank_ic_{_safe_m}_{_safe_ds}_{test_start}_{test_end}.csv",
        )
    if lite_info:
        print("")
        regression_report(
            y_te_aligned,
            test_pred,
            "Test",
            trade_dates=[m[1] for m in meta_te],
            daily_rank_ic_path=_daily_ric_path,
            model_type_label=model_type,
        )
    else:
        print("")
        regression_report(
            y_tr_aligned, train_pred, "Train", trade_dates=[m[1] for m in meta_tr]
        )
        regression_report(
            y_val_aligned, val_pred, "Val", trade_dates=[m[1] for m in meta_val]
        )
        print("")
        regression_report(
            y_te_aligned,
            test_pred,
            "Test",
            trade_dates=[m[1] for m in meta_te],
            daily_rank_ic_path=_daily_ric_path,
            model_type_label=model_type,
        )

    # 回测：仅针对测试集（Base vs TopK rolling）
    backtest_t0 = time.perf_counter() if profile else 0.0
    _backtest_testset_cr_ar_sr_compare(
        meta_te=meta_te,
        y_te=y_te_aligned,
        test_pred=test_pred,
        n_forward=n_forward,
        top_k=top_k,
        topk_pred_threshold=topk_pred_threshold,
        initial_cash=initial_cash,
        annual_trading_days=annual_trading_days,
        lite_info=lite_info,
        data_source=data_source,
        backtest_cost_preset=backtest_cost_preset,
        rebalance_emax=rebalance_emax,
    )

   
    print("")
    #regression_as_class_report(y_tr_aligned, train_pred, "Train")
    #regression_as_class_report(y_val_aligned, val_pred, "Val")
    #regression_as_class_report(y_te_aligned, test_pred, "Test")

    # 将 Train / Val / Test 的预测结果写入 train_db（使用与 pred 同序的 y，保证对齐）
    _write_pred_tables(
        data_source=data_source,
        use_corr_pca=use_corr_pca,
        lite_info=lite_info,
        meta_tr=meta_tr,
        y_tr=y_tr_aligned,
        train_pred=train_pred,
        meta_val=meta_val,
        y_val=y_val_aligned,
        val_pred=val_pred,
        meta_te=meta_te,
        y_te=y_te_aligned,
        test_pred=test_pred,
    )
    if profile:
        _time_section("backtest_and_write_tables", backtest_t0)
        print("[profile] timings(seconds) (sorted desc):")
        for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            print(f"[profile]  {k}: {v:.3f}s")
    return model


def _write_pred_tables(
    *,
    data_source: str,
    use_corr_pca: bool,
    lite_info: bool = False,
    meta_tr: List[Tuple[str, str]],
    y_tr: np.ndarray,
    train_pred: np.ndarray,
    meta_val: List[Tuple[str, str]],
    y_val: np.ndarray,
    val_pred: np.ndarray,
    meta_te: List[Tuple[str, str]],
    y_te: np.ndarray,
    test_pred: np.ndarray,
) -> None:
    """将 Train/Val/Test 的预测与真实收益写入 train_db 的预测表。"""
    def to_df(meta: List[Tuple[str, str]], y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        rows = [
            {"trade_date": m[1], "ts_code": m[0], "predict": float(y_pred[i]), "rev": float(y_true[i])}
            for i, m in enumerate(meta)
        ]
        return pd.DataFrame(rows)

    db_path = os.path.join(get_db_path(), "train_db.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)

    base = f"cs_{data_source}_{'pca' if use_corr_pca else 'plain'}"
    t_train, t_val, t_test = (
        f"{base}_train",
        f"{base}_val",
        f"{base}_test",
    )
    to_df(meta_tr, y_tr, train_pred).to_sql(t_train, conn, if_exists="replace", index=False)
    to_df(meta_val, y_val, val_pred).to_sql(t_val, conn, if_exists="replace", index=False)
    to_df(meta_te, y_te, test_pred).to_sql(t_test, conn, if_exists="replace", index=False)

    conn.close()
    if not lite_info:
        print(f"Prediction tables written: {db_path} | {t_train}, {t_val}, {t_test}")


def _eval_loss_stock_ts(
    model,
    loader,
    loss_fn,
    device,
    *,
    per_batch_mean: bool = False,
    mci_graph: Optional[Any] = None,
):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out, yb = _forward_stock_ts_batch(
                model, batch, device, mci_graph=mci_graph
            )
            b_loss = loss_fn(out, yb).item()
            if per_batch_mean:
                total += b_loss
                n += 1
            else:
                total += b_loss * batch[0].shape[0]
                n += batch[0].shape[0]
    return total / n if n else float("inf")


def _predict_stock_ts(model, loader, device, *, mci_graph: Optional[Any] = None):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            out, _yb = _forward_stock_ts_batch(
                model, batch, device, mci_graph=mci_graph
            )
            preds.append(out.cpu().numpy().ravel())
    return np.concatenate(preds) if preds else np.array([])


def _predict_stock_ts_with_labels(
    model, loader, device, *, mci_graph: Optional[Any] = None
):
    """
    在一次遍历 DataLoader 时同时收集预测值和真实值，保证 pred[i] 与 y_true[i] 严格对应同一样本。
    用于效果统计时避免预测与真实值错位。
    """
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            out, yb = _forward_stock_ts_batch(
                model, batch, device, mci_graph=mci_graph
            )
            preds.append(out.cpu().numpy().ravel())
            labels.append(yb.cpu().numpy().ravel())
    pred_arr = np.concatenate(preds) if preds else np.array([])
    label_arr = np.concatenate(labels) if labels else np.array([])
    return pred_arr, label_arr

