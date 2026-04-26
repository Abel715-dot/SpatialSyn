#!/usr/bin/env python3
"""
Paired t-tests on daily test-set Rank IC: SpatialSyn (or any baseline CSV) vs other models.

Expects CSVs from main.py / train_stock_ts with --daily_rank_ic_dir, columns:
  trade_date, rank_ic, (optional) model_type, pearson_ic
Default filename pattern:
  daily_rank_ic_{model}_{dataset}_{test_start}_{test_end}.csv

For each comparison model, uses inner join on trade_date and tests
  H0: E[IC_baseline - IC_other] = 0
  H1: mean difference > 0  (one-sided, baseline better)

This is a paired t-test on (IC_syn - IC_other) per day; same trading days for both models.
For serial correlation in daily IC, consider Newey–West in a separate analysis; this script
matches the user-requested simple t-test on daily series.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _read_rank_ic_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ("trade_date", "rank_ic")
    for c in need:
        if c not in df.columns:
            raise ValueError(
                f"{path!r} 缺少列 {c!r}，当前为 {list(df.columns)}。请使用 --daily_rank_ic_dir 导出的表。"
            )
    out = df.loc[:, list(need)].copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["rank_ic"] = pd.to_numeric(out["rank_ic"], errors="coerce")
    return out.dropna(subset=["rank_ic"])


def _ttest_paired_greater(
    a: np.ndarray, b: np.ndarray
) -> Tuple[float, float, float, int, float]:
    """
    Paired t-test: test mean(a - b) > 0 vs 0.
    Returns: t_stat, p_two_sided, p_greater, n, mean_diff.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = int(a.shape[0])
    if n < 2:
        return (float("nan"),) * 3 + (n, float("nan"))
    d = a - b
    mean_diff = float(np.mean(d))
    t, p_two = stats.ttest_rel(a, b, nan_policy="omit")
    t, p_two = (float(t), float(p_two))
    if t > 0:
        p_g = p_two / 2.0
    else:
        p_g = 1.0 - p_two / 2.0
    return t, p_two, p_g, n, mean_diff


def _p_value_to_decimal_str(x: Any) -> str:
    """
    固定小数字符串（不用科学计数法）；保留足够小数位以免极小 p 在 CSV 中变成 0。
    """
    if not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return ""
    v = float(x)
    s = f"{v:.15f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _format_float_cell(x: Any) -> str:
    if not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return ""
    s = f"{float(x):.8f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else f"{int(float(x))}"


def write_ttest_results_csv(df: pd.DataFrame, path: str) -> None:
    """
    将 t-test 结果表写入一个 CSV 文件；p 值等浮点列写成普通小数，避免 1.2e-5 形式（便于读表/进 Excel）。
    文本列、整数列原样输出。
    """
    ddir = os.path.dirname(path)
    if ddir:
        os.makedirs(ddir, exist_ok=True)
    out = df.copy()
    p_cols = {c for c in out.columns if str(c).startswith("p_value")}
    for c in out.columns:
        s = out[c]
        if c in p_cols and np.issubdtype(s.dtype, np.floating):
            out[c] = s.map(_p_value_to_decimal_str)
        elif c == "n_overlap_days":
            out[c] = s.map(lambda v: int(v) if pd.notna(v) and np.isfinite(v) else "")
        elif np.issubdtype(s.dtype, np.floating) and c not in p_cols:
            out[c] = s.map(_format_float_cell)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {path} ({len(out)} rows)")


def _parse_rank_ic_filename(path: str) -> Dict[str, str]:
    """
    Parse: daily_rank_ic_{model}_{dataset}_{test_start}_{test_end}.csv
    """
    base = os.path.basename(path)
    m = re.match(
        r"^daily_rank_ic_(.+)_([A-Za-z0-9._-]+)_(\d{8})_(\d{8})\.csv$",
        base,
    )
    if not m:
        raise ValueError(
            f"Unexpected filename format: {base!r}. "
            "Expected daily_rank_ic_{model}_{dataset}_{test_start}_{test_end}.csv"
        )
    return {
        "model": m.group(1),
        "dataset": m.group(2),
        "test_start": m.group(3),
        "test_end": m.group(4),
    }


def run(
    baseline_path: str,
    other_paths: List[str],
    out_csv: Optional[str],
) -> pd.DataFrame:
    base = _read_rank_ic_table(baseline_path)
    base = base.rename(columns={"rank_ic": "rank_ic_baseline"}).drop_duplicates("trade_date")
    rows = []
    bmeta = _parse_rank_ic_filename(baseline_path)
    blabel = bmeta["model"]
    for op in other_paths:
        ometa = _parse_rank_ic_filename(op)
        oth = _read_rank_ic_table(op).rename(columns={"rank_ic": "rank_ic_other"}).drop_duplicates("trade_date")
        merged = base.merge(oth, on="trade_date", how="inner", validate="one_to_one")
        n = len(merged)
        t, p2, p_g, n_eff, mdf = _ttest_paired_greater(
            merged["rank_ic_baseline"].to_numpy(), merged["rank_ic_other"].to_numpy()
        )
        rows.append(
            {
                "baseline": blabel,
                "other": ometa["model"],
                "dataset": bmeta["dataset"],
                "test_start": bmeta["test_start"],
                "test_end": bmeta["test_end"],
                "n_overlap_days": n,
                "mean_rank_ic_baseline": float(merged["rank_ic_baseline"].mean()),
                "mean_rank_ic_other": float(merged["rank_ic_other"].mean()),
                "mean_diff_baseline_minus_other": mdf,
                "t_stat_paired": t,
                "p_value_two_sided": p2,
                "p_value_one_sided_baseline_greater": p_g,
            }
        )
    out = pd.DataFrame(rows)
    if out_csv:
        write_ttest_results_csv(out, out_csv)
    return out


def _discover_others(ic_dir: str, baseline_name_substr: str) -> List[Tuple[str, List[str]]]:
    paths = sorted(glob.glob(os.path.join(ic_dir, "daily_rank_ic_*.csv")))
    if not paths:
        raise SystemExit(f"No daily_rank_ic_*.csv in {ic_dir!r}")
    base_paths = [
        p for p in paths if baseline_name_substr.lower() in os.path.basename(p).lower()
    ]
    if not base_paths:
        raise SystemExit(
            f"No file in {ic_dir!r} contains {baseline_name_substr!r} in its name."
        )
    discovered: List[Tuple[str, List[str]]] = []
    for base_path in sorted(base_paths):
        bmeta = _parse_rank_ic_filename(base_path)
        others: List[str] = []
        for p in paths:
            if os.path.normpath(p) == os.path.normpath(base_path):
                continue
            try:
                meta = _parse_rank_ic_filename(p)
            except ValueError:
                continue
            # Only compare with same dataset and same test period.
            if (
                meta["dataset"] == bmeta["dataset"]
                and meta["test_start"] == bmeta["test_start"]
                and meta["test_end"] == bmeta["test_end"]
            ):
                others.append(p)
        discovered.append((base_path, others))
    return discovered


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Paired t-test: baseline daily Rank IC vs each other model (same trade_date join)."
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to daily_rank_ic_*.csv for the baseline (e.g. SpatialSyn).",
    )
    ap.add_argument(
        "--others",
        type=str,
        nargs="*",
        default=[],
        help="Paths to other model daily_rank_ic CSVs (pairwise vs baseline).",
    )
    ap.add_argument(
        "--ic_dir",
        type=str,
        default="./t_test",
        help="Directory with daily_rank_ic_*.csv (default: ./t_test).",
    )
    ap.add_argument(
        "--ic_dir_auto_baseline",
        type=str,
        default="SpatialSyn",
        help="With --ic_dir, pick as baseline the file whose name contains this substring (default SpatialSyn).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="t_test/rank_ic_pairwise_ttest_results.csv",
        help="Output results CSV (default: t_test/rank_ic_pairwise_ttest_results.csv).",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print the full results table to stdout.",
    )
    ns = ap.parse_args(argv)

    use_manual = bool(ns.baseline) or bool(ns.others)
    if use_manual:
        if not ns.baseline or not ns.others:
            print("Manual mode requires both --baseline and --others.", file=sys.stderr)
            ap.print_help()
            raise SystemExit(1)
        base, others = ns.baseline, list(ns.others)
    else:
        discovered = _discover_others(ns.ic_dir, ns.ic_dir_auto_baseline)
        all_res: List[pd.DataFrame] = []
        for base, others in discovered:
            if not others:
                print(
                    f"Skip baseline without matched peers: {os.path.basename(base)}",
                    file=sys.stderr,
                )
                continue
            all_res.append(run(base, others, out_csv=None))
        if not all_res:
            raise SystemExit(
                f"No comparable other-model files found in {ns.ic_dir!r} for any baseline matched by {ns.ic_dir_auto_baseline!r}."
            )
        res = pd.concat(all_res, ignore_index=True)

    if use_manual:
        res = run(base, others, out_csv=None)
    write_ttest_results_csv(res, ns.out)
    if ns.verbose:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
