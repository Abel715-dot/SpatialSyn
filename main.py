from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any, Dict

import torch
from stock_ts_pred import train_stock_ts

# Merged from former main-informer.py: same CLI flags live here; defaults below follow
# the original main.py only. To reproduce old informer entry defaults, pass e.g.:
#   --no-sse --model_type informer --n_forward 1 --lr 2e-4 --dropout 0.2
#   --backtest-cost auto --rebalance-emax 5
#   --d-model 26 --n-heads 1 --d-ff 52 --no-distil

# Model-local overrides, keyed by model_type (case-insensitive at runtime).
# Rule: for keys present here, local value overrides argparse after parse_args()
# (i.e. it wins over both parser defaults and CLI).
# You can also put model-unique kwargs here; they will be forwarded to build_model(**kwargs).
#
# Common override keys you asked for:
# hidden_size, lr, training_loss, dropout, num_layers, early_stopping,
# min_delta, patience, batch_size, epochs, split_mode
MODEL_LOCAL_PARAM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "SpatialSyn": {
        "use_corr_pca": True,
        "pos_sse": True,
        "neg_sse": False,
        "corr_threshold": 0.3,
        "corr_neg_threshold": 0.1,
        "min_corr_stocks": 3,
        "corr_fallback_mode": "zero",
        "pca_components": 1,
        "features": "open_ret,close_ret,high_ret,low_ret,vol",
        "with_pca_ratio": False,
        "with_pca_self_coef_feature": False,
        "with_corr_count_feature": False,
        "pca_loading_sign_fix": 2,
        "hidden_size": 25,
        "pca_lstm_hidden_size": None,
        "lr": 2e-4,
        "training_loss": "ic",
        "dropout": 0.2,
        "num_layers": 1,
        "early_stopping": True,
        "min_delta": 0.0,
        "patience": 5,
        "batch_size": 256,
        "epochs": 30,
        "split_mode": "time",
    },
    "LSTM": {"lr":2e-4,"training_loss":"ic","num_layers":1,"hidden_size":25,"use_corr_pca":True},
    "GRU": {"lr":2e-4,"training_loss":"mse","num_layers":1,"hidden_size":25 },
    "informer": {
        # Align with thuml Autoformer/Informer official run.py defaults.
        # Informer/Autoformer shared params (former global CLI defaults, now generic CLI):
        # d_model=64, n_heads=8, enc_layers=1, dec_layers=1, d_ff=128,
        # factor=3, distil=False, moving_avg=5, seq_len=None.
        "d_model": 64,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 128,
        "factor": 3,
        "distil": False,
        "moving_avg": 5,
        "seq_len": None,
        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "training_loss": "mse",
        "lr": 2e-4,
    },
    "autoformer": {
        # Align with thuml Autoformer official run.py defaults.
        # Informer/Autoformer shared params (former global CLI defaults, now generic CLI):
        # d_model=64, n_heads=8, enc_layers=1, dec_layers=1, d_ff=128,
        # factor=3, distil=False, moving_avg=5, seq_len=None.
        "d_model": 64,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 128,
        "factor": 3,
        "moving_avg": 5,
        "seq_len": None,
        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "training_loss": "mse",
        "lr": 2e-4,
    },
    "mamba": {
        "hidden_size": 32,
        "num_layers": 1,
         "dropout": 0.2,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "training_loss": "mse",
        "lr": 1e-4,   
    },
    "thgnn": {
        # QuantBench/q4l defaults: THGNN spatial + GRU temporal.
        "hidden_size": 256,
        "num_layers": 2,
        #"dropout": 0.0,
        "n_heads": 8,
        "out_features": 32,
        # Paper-style objective: daily top/bottom-k binary classification (BCE).
        "training_loss": "bce",
        "bce_top_k": 100,
        "bce_bottom_k": 100,
        "lr": 2e-4,
        #"epochs": 100,
    },
    "transformer": {},
    "crossformer": {
        # Official Crossformer defaults from main_crossformer.py
        "d_model": 64,
        "n_heads": 2,
        "e_layers": 1,
        "d_ff": 128,
        "factor": 10,
        "dropout": 0.2,
        "seq_len": 15,
        "training_loss": "mse",
        "lr": 1e-4,
        #"epochs": 20,
        "batch_size": 32,
        #"early_stopping": True,
        #"patience": 3,
        "target_feature_index": 1,
        "seg_len": 5,
        "win_size": 2,
        "baseline": False,
    },
    "itransformer": {
        # Align with official run.py parser defaults.
        "d_model": 64,
        "e_layers": 1,
        "d_ff": 128,
        #"dropout": 0.1,
        "seq_len": 15,
        "training_loss": "mse",
        "lr": 1e-4,
        #"epochs": 10,
        "batch_size": 32,
        #"early_stopping": True,
        #"patience": 3,
        "target_feature_index": 1,
        "factor": 3,
        "output_attention": False,
        "use_norm": True,
    },
    "patchtst": {
        # Official PatchTST ETTh1 baseline script.
        "d_model": 16,
        "n_heads": 4,
        "e_layers": 1,
        "d_ff": 128,
        "dropout": 0.3,
        "seq_len": 15,
        "training_loss": "mse",
        "lr": 1e-4,
        #"epochs": 100,
        "batch_size": 128,
        #"early_stopping": True,
        #"patience": 100,
        "patch_len": 16,
        "stride": 8,
        "fc_dropout": 0.3,
        "head_dropout": 0.0,
        "target_feature_index": 1,
    },
    "master": {
        # Official MASTER defaults (main.py + base_model.py).
        # MASTER does not use validation early-stop in official code;
        # it uses MSE and a train-loss threshold condition.
        "d_model": 64,
        #"dropout": 0.5,
        "master_d_feat": 5,
        "master_t_nhead": 4,
        "master_s_nhead": 2,
        "master_beta": 5.0,
        "master_gate_input_start_index": 5,
        "master_gate_input_end_index": 10,
        "training_loss": "mse+",
        "lr": 1e-5,
        #"epochs": 1,
        #"early_stopping": False,
    },
    "deltalag": {
        # DeltaLag paper (ICAIF 2025): cross-asset lead-lag + monotonic ranking loss.
        "hidden_size": 64,
        #"dropout": 0.1,
        "deltalag_max_lag": 5,
        # Paper experiments use k=2 leaders (top-k pairs).
        "deltalag_top_k": 2,
        # DeltaLag paper uses ranking-oriented objective for cross-sectional selection.
        "training_loss": "mon",
        "lr": 1e-4,
    },
    "dtml": {
        # QuantBench official dtml.yaml + base_spatial_model.
        "node_emb_dim": 64,
        "out_features": 64,
        "num_heads": 8,
        "num_layers": 1,
        "beta": 0,
        #"dropout": 0.1,
        # Paper-style objective: movement-direction BCE (return sign -> 0/1).
        "training_loss": "bce",
        "lr": 2e-4,
    },
    "mci-gru": {
        # WinstonLiyt/MCI-GRU code/csi300.py：StockPredictionModel 超参 + 训练脚本核心项。
        # 训练管线：按日截面 batch + 相关矩阵构图（见 stock_ts_pred）。
        "hidden_size": 32,
        "hidden_size_gat1": 5,
        "output_gat1": 32,
        "gat_out_channels": 4,
        "gat_heads": 4,
        "hidden_size_gat2": 5,
        "num_hidden_states": 16,
        "embed_dim": 32,
        "mci_judge_value": 0.8,
        "mci_corr_lookback": 250,
        "training_loss": "mse",
        "lr": 1e-3,
        "epochs": 20,
        "early_stopping": False,
    },
}
MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn_Pro"] = {
    **dict(MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn"]),
    "with_pca_ratio": False,
    "with_pca_self_coef_feature": False,
    "with_corr_count_feature": True,
    "pca_lstm_hidden_size": 32
}
MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn_LN"] = {
    **dict(MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn"]),
    "use_fusion_layernorm": True,
}
MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn_Pro_LN"] = {
    **dict(MODEL_LOCAL_PARAM_OVERRIDES["SpatialSyn_Pro"]),
    "use_fusion_layernorm": True,
}


def _non_negative_int(value: str) -> int:
    iv = int(value)
    if iv < 0:
        raise argparse.ArgumentTypeError(f"min_corr_stocks must be >= 0, got {iv}")
    return iv


def _pca_loading_sign_fix_int(value: str) -> int:
    iv = int(value)
    if iv not in (0, 1, 2):
        raise argparse.ArgumentTypeError(f"pca_loading_sign_fix must be 0, 1, or 2, got {iv}")
    return iv


def _get_model_local_overrides(model_type: str) -> Dict[str, Any]:
    mt = str(model_type)
    if mt in MODEL_LOCAL_PARAM_OVERRIDES:
        return dict(MODEL_LOCAL_PARAM_OVERRIDES[mt])
    mt_lower = mt.lower()
    for k, v in MODEL_LOCAL_PARAM_OVERRIDES.items():
        if str(k).lower() == mt_lower:
            return dict(v)
    return {}


def _collect_explicit_known_cli_dests(
    parser: argparse.ArgumentParser, raw_tokens: list[str]
) -> set[str]:
    """
    Collect parser destination names explicitly provided via known CLI options.
    """
    option_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    explicit: set[str] = set()
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue

        key = tok.split("=", 1)[0]
        if key in option_to_dest:
            explicit.add(option_to_dest[key])
            # for `--key value`, skip next value token
            if "=" not in tok and i + 1 < len(raw_tokens) and not raw_tokens[i + 1].startswith("--"):
                i += 2
                continue
        i += 1
    return explicit


def apply_model_local_param_overrides(
    args: argparse.Namespace, cli_specified_dests: set[str] | None = None
) -> Dict[str, Any]:
    """
    Apply ``MODEL_LOCAL_PARAM_OVERRIDES`` for ``args.model_type``.

    Runs **after** ``ArgumentParser.parse_args``, which has already merged
    parser defaults with CLI. Local overrides act as model defaults only:
    if a key is explicitly passed via CLI, that CLI value is preserved.

    If the local dict contains ``model_type``, that value is applied first, then
    the preset for the **resulting** ``model_type`` is loaded and merged, with
    keys from the originally selected preset (except ``model_type``) winning on
    conflict — so you can point one bucket at another model and still overlay
    keys in the first bucket.

    Keys in the local dict that are **not** argparse destinations (e.g. model-only
    kwargs for ``build_model``) are returned as ``model_extra_kwargs`` and are not
    set on ``args``. Use the same names as ``dest`` (e.g. ``use_corr_pca``, not
    ``sse``).
    """
    primary = _get_model_local_overrides(args.model_type)
    if not primary:
        return {}

    local = dict(primary)
    cli_specified_dests = cli_specified_dests or set()
    ns = vars(args)
    known_dest = set(ns.keys())

    if "model_type" in local:
        new_mt = local.pop("model_type")
        if "model_type" in known_dest:
            ns["model_type"] = new_mt
        base = _get_model_local_overrides(ns["model_type"])
        local = {**base, **local}

    for k, v in local.items():
        if k in known_dest and k not in cli_specified_dests:
            ns[k] = v
    return {k: v for k, v in local.items() if k not in known_dest}


def _parse_bool_text(value: str) -> bool:
    lv = str(value).strip().lower()
    if lv in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lv in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _coerce_cli_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return s
    ls = s.lower()
    if ls in {"none", "null"}:
        return None
    if ls in {"true", "false", "yes", "no", "on", "off"}:
        try:
            return _parse_bool_text(s)
        except ValueError:
            pass
    try:
        if "." not in s and "e" not in ls:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _coerce_for_param(param_name: str, value: Any, sig: inspect.Signature) -> Any:
    if param_name not in sig.parameters:
        return _coerce_cli_scalar(value)
    param = sig.parameters[param_name]
    default = param.default
    if default is inspect._empty:
        return _coerce_cli_scalar(value)
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return _parse_bool_text(str(value))
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool):
            return int(value)
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    return _coerce_cli_scalar(value)


def parse_generic_cli_kwargs(
    unknown_tokens: list[str], train_fn_sig: inspect.Signature
) -> Dict[str, Any]:
    """
    Parse unknown CLI flags into key/value pairs.

    Supports:
    - --key value
    - --key=value
    - --key            (boolean True)
    - --no-key         (boolean False)
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown_tokens):
        tok = unknown_tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected token {tok!r}; expected --<name>")

        key: str
        value: Any

        if tok.startswith("--no-"):
            key = tok[5:]
            value = False
            i += 1
        elif "=" in tok:
            key, raw = tok[2:].split("=", 1)
            value = raw
            i += 1
        else:
            key = tok[2:]
            if i + 1 < len(unknown_tokens) and not unknown_tokens[i + 1].startswith("--"):
                value = unknown_tokens[i + 1]
                i += 2
            else:
                value = True
                i += 1

        if not key:
            raise ValueError(f"Invalid option name from token: {tok!r}")
        key = key.replace("-", "_")
        kwargs[key] = _coerce_for_param(key, value, train_fn_sig)
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entry point (plain). Defaults match stock_ts_pred.py"
    )
    # ---- Most common knobs (kept at top) ----
    p.add_argument("--data_source", type=str, default="hs300", choices=["hs300", "sp500"])

    p.add_argument("--use_corr_pca", dest="use_corr_pca", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pos_sse", dest="pos_sse", action=argparse.BooleanOptionalAction, default=True, help="include positive pool in model input (default on); if off, those columns are not concatenated; sample cache unchanged")
    p.add_argument("--neg_sse", dest="neg_sse", action=argparse.BooleanOptionalAction, default=False, help="include negative pool in model input (default off); if off, not concatenated; sample cache unchanged")
    p.add_argument("--corr_threshold", type=float, default=0.3)
    p.add_argument("--corr_neg_threshold", type=float, default=0.1, dest="corr_neg_threshold", help="negative pool: include stocks with corr <= -this value (independent of --corr_threshold)")
    p.add_argument("--min_corr_stocks", type=_non_negative_int, default=3, help="minimum correlated stocks threshold (must be >=0)")
    p.add_argument("--corr_fallback_mode", type=str, default="zero", choices=["zero", "drop"], help="when corr stock count < min_corr_stocks: zero PCA features or drop sample")
    p.add_argument("--pca_components", type=int, default=1, dest="pca_components")
    p.add_argument("--features", type=str, default="open_ret,close_ret,high_ret,low_ret,vol")
    p.add_argument("--with_pca_ratio", dest="with_pca_ratio", action=argparse.BooleanOptionalAction, default=False, help="whether to append explained_variance_ratio_ as features")
    p.add_argument("--with_pca_self_coef_feature", dest="with_pca_self_coef_feature", action=argparse.BooleanOptionalAction, default=False, help="whether to append positive-pool self-stock coefficients in PCA component vectors as features")
    p.add_argument("--with_corr_count_feature", dest="with_corr_count_feature", action=argparse.BooleanOptionalAction, default=False, help="whether to append number of corr>=threshold stocks as a feature")
    p.add_argument("--pca_loading_sign_fix", type=_pca_loading_sign_fix_int, default=2, dest="pca_loading_sign_fix", help="PCA loading sign: 0=off, 1=sum(loadings)<0 (tie: max|coef|), 2=count(neg)>count(pos), tie uses rule 1")

    
    # ---- Early stopping knobs (just under common ones) ----
    p.add_argument("--model_type", type=str, default="LSTM", choices=["SpatialSyn", "LSTM", "GRU", "informer", "autoformer", "mamba", "thgnn", "transformer", "dtml", "crossformer", "itransformer", "patchtst", "master", "deltalag", "mci-gru", ])
    p.add_argument("--hidden_size", type=int, default=25)
    p.add_argument("--pca_lstm_hidden_size", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--Ts", type=int, default=15, help="sequence window length")
    p.add_argument("--Tw", type=int, default=40, help="correlated stocks observation window (Tw > Ts)")
    p.add_argument("--n_forward", type=int, default=1, choices=[1, 2, 3, 4, 5])

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--early_stopping", dest="early_stopping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min_delta", type=float, default=0, dest="min_delta")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="override device selection") 
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--training_loss", type=str, default="mse", choices=["mse", "mse+", "ic", "mon", "bce"], help="mse (normal mini-batch), mse+ (MSE/day-batch), ic (IC/day-batch), mon (DeltaLag monotonic ranking loss), or bce (daily top/bottom-k BCE)")
    p.add_argument("--bce_top_k", type=int, default=100, dest="bce_top_k", help="for training_loss=bce: daily top-k labeled as class 1")
    p.add_argument("--bce_bottom_k", type=int, default=100, dest="bce_bottom_k", help="for training_loss=bce: daily bottom-k labeled as class 0")

    p.add_argument("--train_start", type=str, default="20160101", help="training start YYYYMMDD")
    p.add_argument("--train_end", type=str, default="20260320", help="panel end YYYYMMDD")
    p.add_argument("--test_start", type=str, default="20240101", help="independent test start YYYYMMDD")
    p.add_argument("--test_end", type=str, default="20251231", help="independent test end YYYYMMDD")
    p.add_argument("--split_mode", type=str, default="time", choices=["random", "time"])

    p.add_argument("--top_k", type=int, default=20, help="backtest optimization: top-k stocks by prediction")
    p.add_argument("--topk_pred_threshold", type=float, default=-1000, help="backtest optimization: only predict > threshold enters top-k ranking")
    p.add_argument("--backtest_cost_preset", type=str, default="auto", choices=["none", "auto", "cn", "us", "hk", "uk", "jp"], dest="backtest_cost_preset", help="Backtest fees: none; auto (cn/us from data_source); cn=A-share itemized; us=SEC+TAF on sells; hk/uk/jp=legacy %%. If not none, logs both ex-fee and fee-adjusted CR/AR/SR")
    p.add_argument("--rebalance_emax", type=int, default=None, help="Opt: per rebalance at most Emax sells/buys (non-ideal held / ideal not held); omit for unlimited. Applies only to fee-adjusted Opt; ex-fee Opt always uses unlimited rebalancing")
    p.add_argument("--initial_cash", type=float, default=1_000_000.0, dest="initial_cash")
    p.add_argument("--annual_trading_days", type=int, default=252, dest="annual_trading_days")

   
    p.add_argument("--cache", dest="use_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lite_info", dest="lite_info", action=argparse.BooleanOptionalAction, default=False, help="only print minimal info for effect comparison")
    # Backtest capitalization defaults should come from main.py, not train_stock_ts.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--daily_rank_ic_dir",
        type=str,
        default=".",
        help="if set, write test-set daily cross-sectional Rank IC to this dir's t_test subfolder "
        "(file: daily_rank_ic_{model_type}_{data_source}_{test_start}_{test_end}.csv)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    raw_tokens = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, unknown_tokens = parser.parse_known_args(argv)
    train_sig = inspect.signature(train_stock_ts)
    generic_cli_kwargs = parse_generic_cli_kwargs(unknown_tokens, train_sig)
    explicit_known_cli_dests = _collect_explicit_known_cli_dests(parser, raw_tokens)
    model_extra_kwargs = apply_model_local_param_overrides(
        args, cli_specified_dests=explicit_known_cli_dests
    )
    train_param_names = set(train_sig.parameters.keys())
    train_param_names.discard("model_extra_kwargs")
    generic_train_kwargs = {
        k: v
        for k, v in generic_cli_kwargs.items()
        if k in train_param_names
    }
    generic_model_extra_kwargs = {
        k: v
        for k, v in generic_cli_kwargs.items()
        if k not in train_param_names
    }
    model_extra_kwargs.update(generic_model_extra_kwargs)

    model_upper = args.model_type.upper()
    device = None
    if args.device != "auto":
        # Explicit override always wins.
        device = torch.device(args.device)
    else:
        # Auto policy by model type:
        # - GRU: prefer CPU (observed faster than MPS in this project).
        # - LSTM / Transformer-family: prefer MPS on Apple Silicon.
        if model_upper == "GRU" and args.hidden_size <32:
            device = torch.device("cpu")
        elif model_upper == "MAMBA":
            # CUDA: fused mamba-ssm; Apple Silicon: pure MambaTorch in mamba_torch.py.
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif model_upper == "THGNN":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif model_upper in ("SPATIALSYN", "SPATIALSYN_PRO", "SPATIALSYN_LN", "SPATIALSYN_PRO_LN"):
            # SpatialSyn family: dual LSTM branches — CUDA, else MPS on Apple Silicon, else CPU.
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            # Any other model type (includes Crossformer: MPS-safe contiguous patches in third_party Crossformer).
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

    features = None
    if isinstance(args.features, str) and args.features.strip():
        features = [s.strip() for s in args.features.split(",") if s.strip()]

    train_kwargs: Dict[str, Any] = dict(
        data_source=args.data_source,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        Ts=args.Ts,
        Tw=args.Tw,
        n_forward=args.n_forward,
        corr_threshold=args.corr_threshold,
        corr_neg_threshold=args.corr_neg_threshold,
        min_corr_stocks=args.min_corr_stocks,
        corr_fallback_mode=args.corr_fallback_mode,
        pca_components=args.pca_components,
        pca_loading_sign_fix=args.pca_loading_sign_fix,
        with_pca_ratio=args.with_pca_ratio,
        with_pca_self_coef_feature=args.with_pca_self_coef_feature,
        with_corr_count_feature=args.with_corr_count_feature,
        use_corr_pca=args.use_corr_pca,
        pos_sse=args.pos_sse,
        neg_sse=args.neg_sse,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        pca_lstm_hidden_size=args.pca_lstm_hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        use_cache=args.use_cache,
        device=device,
        features=features,
        split_mode=args.split_mode,
        top_k=args.top_k,
        topk_pred_threshold=args.topk_pred_threshold,
        backtest_cost_preset=args.backtest_cost_preset,
        rebalance_emax=args.rebalance_emax,
        lite_info=args.lite_info,
        training_loss=args.training_loss,
        bce_top_k=args.bce_top_k,
        bce_bottom_k=args.bce_bottom_k,
        initial_cash=args.initial_cash,
        annual_trading_days=args.annual_trading_days,
        daily_rank_ic_dir=args.daily_rank_ic_dir,
        model_extra_kwargs=model_extra_kwargs,
    )
    train_kwargs.update(generic_train_kwargs)
    train_stock_ts(**train_kwargs)


if __name__ == "__main__":
    main()
