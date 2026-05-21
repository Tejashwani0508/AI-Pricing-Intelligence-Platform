"""
Demand Forecasting Engine Module

Predicts future demand for each product using Facebook Prophet
for time-series forecasting. Supports multiple products simultaneously
with synthetic historical data generation when actual time-series
data is not available.

Architecture:
  Input DataFrame (product sales) → Generate time-series → Prophet model → Forecast

Key Functions:
  - train_model()        — Fit a Prophet model on a product's history
  - forecast_demand()    — Generate future predictions for one product
  - evaluate_model()     — Compute error metrics on holdout data
  - batch_forecast()     — Run forecasting for all products in parallel

Output per product:
  - predicted_demand     — Forecasted sales volume for next 30 days
  - forecast_confidence  — Confidence score (0-1) based on model fit
  - forecast_ci_lower    — Lower bound of prediction interval
  - forecast_ci_upper    — Upper bound of prediction interval
  - demand_trend         — Detected trend direction
  - weekly_breakdown     — Week-by-week forecast

Uses Prophet for robust time-series decomposition (trend, weekly seasonality).
Falls back to synthetic history generation when only aggregate sales data is available.
"""

import logging
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Prophet
warnings.filterwarnings("ignore", category=UserWarning, module="prophet")
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None  # type: ignore
    PROPHET_AVAILABLE = False

try:
    import cmdstanpy
    CMDSTANPY_AVAILABLE = True
except ImportError:
    cmdstanpy = None  # type: ignore
    CMDSTANPY_AVAILABLE = False

from utils.config import AppConfig
from utils.helpers import safe_divide, round_half_up

logger = logging.getLogger("ai_pricing.demand_forecaster")


# ─── Constants ─────────────────────────────────────────────────────────────

# Default Prophet parameters
DEFAULT_PROPHET_PARAMS: Dict[str, Any] = {
    "seasonality_mode": "additive",
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "yearly_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "uncertainty_samples": 1000,
    "interval_width": 0.95,
}

# Forecasting horizon
FORECAST_DAYS: int = 30
HISTORY_DAYS: int = 90  # Generate 90 days of history if not provided

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.65

# Model directory
MODEL_DIR_NAME: str = "demand_models"

_PROPHET_BACKEND_OK: Optional[bool] = None
_PROPHET_BACKEND_ERROR: str = ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float with a fallback."""
    try:
        result = float(value)
        if np.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def _align_arrays(*arrays: Any) -> Tuple[np.ndarray, ...]:
    """
    Trim arrays to a common length before vector operations.

    This protects the forecasting code from off-by-one date generation,
    Prophet output drift, missing rows, and inconsistent uploaded histories.
    """
    converted = [np.asarray(arr) for arr in arrays]
    if not converted:
        return tuple()

    common_len = min(len(arr) for arr in converted)
    if common_len <= 0:
        return tuple(arr[:0] for arr in converted)

    if any(len(arr) != common_len for arr in converted):
        logger.warning(
            "Aligning forecast arrays to common length=%s from shapes=%s",
            common_len,
            [arr.shape for arr in converted],
        )
    return tuple(arr[:common_len] for arr in converted)


def _fallback_forecast_result(
    product_data: Dict[str, Any],
    error_message: str = "",
    confidence: float = 0.50,
) -> "ForecastResult":
    """Create an enterprise-safe fallback forecast for one product."""
    sales_volume = max(_safe_float(product_data.get("sales_volume", 0), 0.0), 0.0)
    return ForecastResult(
        product_id=str(product_data.get("product_id", "")),
        product_name=str(product_data.get("product_name", "")),
        predicted_demand=round(sales_volume, 1),
        forecast_confidence=confidence,
        forecast_ci_lower=round(sales_volume * 0.8, 1),
        forecast_ci_upper=round(sales_volume * 1.2, 1),
        demand_trend="stable",
        weekly_forecast={
            "week_1": round(sales_volume / 4, 1),
            "week_2": round(sales_volume / 4, 1),
            "week_3": round(sales_volume / 4, 1),
            "week_4": round(sales_volume / 4, 1),
        },
        seasonality_strength=0.0,
        error_message=error_message,
    )


def get_dependency_versions() -> Dict[str, str]:
    """
    Return installed Prophet/CmdStanPy versions for diagnostics.

    This is safe to call from Streamlit or a terminal command even when one
    dependency is missing.
    """
    versions: Dict[str, str] = {}
    for package_name in ("prophet", "cmdstanpy"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def log_dependency_diagnostics() -> Dict[str, str]:
    """Log Prophet, CmdStanPy, and CmdStan runtime state."""
    versions = get_dependency_versions()
    logger.info(
        "Forecast dependency versions: prophet=%s cmdstanpy=%s",
        versions.get("prophet"),
        versions.get("cmdstanpy"),
    )

    if CMDSTANPY_AVAILABLE and cmdstanpy is not None:
        try:
            cmdstan_path = cmdstanpy.cmdstan_path()
            logger.info("CmdStan path: %s", cmdstan_path)
            versions["cmdstan_path"] = str(cmdstan_path)
        except Exception as exc:
            logger.warning("CmdStan path is unavailable or invalid: %s", exc)
            versions["cmdstan_path"] = f"unavailable: {exc}"
    else:
        logger.warning("cmdstanpy is not installed; Prophet CMDSTANPY backend cannot run.")
        versions["cmdstan_path"] = "cmdstanpy not installed"

    return versions


def _initialize_prophet(params: Dict[str, Any]) -> Prophet:
    """
    Initialize Prophet with a production-safe Stan backend strategy.

    Prophet 1.1.x on Windows can fail with:
    AttributeError: 'Prophet' object has no attribute 'stan_backend'
    when CmdStan is missing or Prophet's bundled cmdstan directory is corrupt.
    We first request CMDSTANPY explicitly so the real backend error is visible,
    then retry the default loader once, and raise a clear RuntimeError if both
    paths are unusable.
    """
    if not PROPHET_AVAILABLE or Prophet is None:
        raise RuntimeError("Prophet is not installed.")

    versions = log_dependency_diagnostics()
    errors: List[str] = []

    try:
        logger.debug("Initializing Prophet with stan_backend='CMDSTANPY'.")
        return Prophet(**params, stan_backend="CMDSTANPY")
    except Exception as exc:
        errors.append(f"CMDSTANPY backend failed: {type(exc).__name__}: {exc}")
        logger.warning(
            "Prophet CMDSTANPY backend initialization failed. "
            "prophet=%s cmdstanpy=%s error=%s",
            versions.get("prophet"),
            versions.get("cmdstanpy"),
            exc,
            exc_info=True,
        )

    try:
        logger.debug("Retrying Prophet initialization with default backend loader.")
        model = Prophet(**params)
        if not hasattr(model, "stan_backend"):
            raise AttributeError(
                "Prophet default loader returned a model without stan_backend."
            )
        return model
    except Exception as exc:
        errors.append(f"default backend failed: {type(exc).__name__}: {exc}")
        logger.warning(
            "Prophet default backend initialization failed; forecasts will use fallback. "
            "error=%s",
            exc,
            exc_info=True,
        )

    raise RuntimeError(
        "Prophet Stan backend unavailable. "
        "This usually means CmdStan is missing/corrupt or Prophet was installed "
        f"with incompatible backend files. Diagnostics={versions}. "
        f"Attempts: {' | '.join(errors)}"
    )


def _prophet_backend_available(force_refresh: bool = False) -> bool:
    """
    Check Prophet backend health once per process.

    Batch forecasting uses this to avoid hundreds of repeated Stan backend
    failures when the environment is broken.
    """
    global _PROPHET_BACKEND_OK, _PROPHET_BACKEND_ERROR

    if _PROPHET_BACKEND_OK is not None and not force_refresh:
        return _PROPHET_BACKEND_OK

    if not PROPHET_AVAILABLE:
        _PROPHET_BACKEND_OK = False
        _PROPHET_BACKEND_ERROR = "Prophet is not installed."
        return False

    try:
        model = _initialize_prophet(DEFAULT_PROPHET_PARAMS)
        _PROPHET_BACKEND_OK = hasattr(model, "stan_backend")
        _PROPHET_BACKEND_ERROR = "" if _PROPHET_BACKEND_OK else "Prophet has no stan_backend."
    except Exception as exc:
        _PROPHET_BACKEND_OK = False
        _PROPHET_BACKEND_ERROR = str(exc)
        logger.warning("Prophet backend health check failed: %s", exc)

    return bool(_PROPHET_BACKEND_OK)


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """
    Forecast result for a single product.
    """
    product_id: str = ""
    product_name: str = ""
    predicted_demand: float = 0.0
    forecast_confidence: float = 0.0
    forecast_ci_lower: float = 0.0
    forecast_ci_upper: float = 0.0
    demand_trend: str = ""          # "increasing", "stable", "declining"
    weekly_forecast: Dict[str, float] = field(default_factory=dict)
    seasonality_strength: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "predicted_demand": round(self.predicted_demand, 2),
            "forecast_confidence": round(self.forecast_confidence, 3),
            "forecast_ci_lower": round(self.forecast_ci_lower, 2),
            "forecast_ci_upper": round(self.forecast_ci_upper, 2),
            "demand_trend": self.demand_trend,
            "weekly_forecast": self.weekly_forecast,
            "seasonality_strength": round(self.seasonality_strength, 3),
        }


@dataclass
class ForecastingReport:
    """
    Aggregate report for a batch forecasting run.
    """
    total_products: int = 0
    forecasted: int = 0
    errors: int = 0
    error_ids: List[str] = field(default_factory=list)

    avg_predicted_demand: float = 0.0
    avg_confidence: float = 0.0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0

    trend_distribution: Dict[str, int] = field(default_factory=lambda: {
        "increasing": 0, "stable": 0, "declining": 0,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_products,
            "forecasted": self.forecasted,
            "errors": self.errors,
            "avg_demand": round(self.avg_predicted_demand, 1),
            "avg_confidence": round(self.avg_confidence, 3),
            "high_confidence": self.high_confidence_count,
            "medium_confidence": self.medium_confidence_count,
            "low_confidence": self.low_confidence_count,
            "trends": self.trend_distribution,
        }

    def summary(self) -> str:
        return (
            f"  Products:  {self.total_products}\n"
            f"  Forecasted: {self.forecasted}\n"
            f"  Errors:    {self.errors}\n"
            f"  Avg demand: {self.avg_predicted_demand:.1f}\n"
            f"  Avg confidence: {self.avg_confidence:.2f}\n"
            f"  Trends: {self.trend_distribution}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _generate_historical_demand(
    base_sales: float,
    demand_trend: float,
    days: int = HISTORY_DAYS,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic daily historical demand data for a product.

    Creates realistic daily sales data with:
    - Base level from aggregate sales_volume
    - Trend component from demand_trend
    - Weekly seasonality (weekday patterns)
    - Random noise

    Args:
        base_sales: Aggregate monthly sales volume.
        demand_trend: Demand trend signal (0-1).
        days: Number of historical days to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 'ds' (date) and 'y' (sales) columns.
    """
    if seed is not None:
        np.random.seed(seed)

    end_date = datetime.now().normalize() if hasattr(datetime.now(), "normalize") else datetime.now()
    dates = pd.date_range(end=end_date, periods=max(int(days), 1), freq="D")

    # Base daily sales (monthly / 30)
    base_daily = base_sales / 30.0 if base_sales > 0 else 3.0

    # Trend: scale demand_trend (0-1) to an annual growth rate
    # demand_trend=0.5 → stable; 1.0 → +20% over period; 0.0 → -20%
    trend_scale = (demand_trend - 0.5) * 0.4
    trend_factor = np.linspace(1.0 - abs(trend_scale), 1.0 + abs(trend_scale), len(dates))

    # Weekly seasonality: weekend boost, mid-week dip
    day_of_week = dates.dayofweek.values
    weekly_pattern = np.where(
        day_of_week >= 5, 1.25,  # weekend +25%
        np.where(day_of_week == 0, 1.10,  # Monday +10%
        np.where(day_of_week == 3, 0.85,  # Thursday -15%
                 1.0)),
    )

    # Random noise (CV ~15%)
    noise = np.random.normal(loc=0, scale=base_daily * 0.15, size=len(dates))

    trend_factor, weekly_pattern, noise = _align_arrays(
        trend_factor, weekly_pattern, noise
    )
    dates = dates[: len(trend_factor)]

    logger.debug(
        "Synthetic demand shapes: dates=%s trend=%s weekly=%s noise=%s",
        len(dates),
        trend_factor.shape,
        weekly_pattern.shape,
        noise.shape,
    )

    daily_sales = base_daily * trend_factor * weekly_pattern + noise
    daily_sales = np.maximum(daily_sales, 0)  # No negative sales

    df = pd.DataFrame({
        "ds": dates,
        "y": daily_sales.round(0),
    })
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE PRODUCT FORECASTING
# ═══════════════════════════════════════════════════════════════════════════

def _compute_confidence_from_model(model: Prophet,
                                    future: pd.DataFrame,
                                    forecast: pd.DataFrame) -> float:
    """
    Compute a confidence score (0-1) based on the forecast uncertainty.

    Uses the coefficient of variation of the prediction interval width
    relative to the forecasted value. Lower uncertainty → higher confidence.

    Args:
        model: Fitted Prophet model.
        future: Future dates DataFrame.
        forecast: Forecast result from model.predict().

    Returns:
        Confidence score between 0 and 1.
    """
    # Extract prediction interval half-width
    half_width = (forecast["yhat_upper"] - forecast["yhat_lower"]).values / 2.0
    yhat = forecast["yhat"].values
    half_width, yhat = _align_arrays(half_width, yhat)
    if len(yhat) == 0:
        return 0.5

    # Compute CV of uncertainty
    mean_width = np.mean(half_width)
    mean_forecast = np.mean(np.abs(yhat))

    if mean_forecast < 0.01:
        return 0.5  # Neutral confidence for near-zero forecasts

    cv = mean_width / mean_forecast
    # Map CV to confidence: CV=0 → 1.0, CV=0.5 → 0.75, CV=1.0 → 0.5
    confidence = 1.0 - min(cv, 1.0) * 0.5
    return np.clip(confidence, 0.0, 1.0)


def _detect_trend(forecast: pd.DataFrame) -> str:
    """
    Detect whether the forecast trend is increasing, stable, or declining.

    Args:
        forecast: Prophet forecast DataFrame with 'trend' column.

    Returns:
        Trend direction string.
    """
    if "trend" not in forecast.columns:
        return "stable"

    trend_values = forecast["trend"].values
    if len(trend_values) < 2:
        return "stable"

    # Compare start vs end of forecast period
    start_trend = trend_values[0]
    end_trend = trend_values[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        change_pct = (end_trend - start_trend) / abs(start_trend) if abs(start_trend) > 0.01 else 0.0

    if change_pct > 0.05:
        return "increasing"
    elif change_pct < -0.05:
        return "declining"
    return "stable"


def _compute_seasonality_strength(model: Prophet, forecast: pd.DataFrame) -> float:
    """
    Estimate the strength of weekly seasonality in the forecast.

    Args:
        model: Fitted Prophet model.
        forecast: Forecast DataFrame.

    Returns:
        Seasonality strength (0 = none, 1 = very strong).
    """
    if "weekly" not in forecast.columns:
        return 0.0

    weekly = forecast["weekly"].values
    yhat = forecast["yhat"].values if "yhat" in forecast.columns else np.array([])
    weekly, yhat = _align_arrays(weekly, yhat)
    if len(weekly) < 2 or np.std(weekly) < 0.01:
        return 0.0

    amplitude = np.max(weekly) - np.min(weekly)
    mean_value = np.mean(yhat)
    if mean_value < 0.01:
        return 0.0

    strength = min(amplitude / mean_value, 1.0)
    return strength


def train_model(
    historical_df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Prophet, Dict[str, float]]:
    """
    Train a Prophet model on historical demand data.

    Args:
        historical_df: DataFrame with 'ds' (datetime) and 'y' (sales) columns.
        params: Prophet parameters (uses defaults if None).

    Returns:
        Tuple of (fitted Prophet model, training metrics dict).

    Raises:
        ValueError: If Prophet is not installed or data is invalid.
        RuntimeError: If model training fails.
    """
    if not PROPHET_AVAILABLE:
        raise ValueError(
            "Prophet is not installed. Install with: pip install prophet pystan"
        )

    if historical_df.empty or "ds" not in historical_df.columns or "y" not in historical_df.columns:
        raise ValueError("Historical data must contain 'ds' and 'y' columns.")

    if params is None:
        params = DEFAULT_PROPHET_PARAMS

    logger.debug(
        f"Training Prophet model on {len(historical_df)} data points"
    )

    try:
        model = _initialize_prophet(params)
        model.fit(historical_df)

        # Simple training metrics
        forecast = model.predict(historical_df)
        actual, predicted = _align_arrays(
            historical_df["y"].values,
            forecast["yhat"].values,
        )
        if len(actual) == 0:
            raise RuntimeError("Prophet produced no aligned training predictions.")
        residuals = actual - predicted
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # Normalised metrics
        mean_actual = float(np.mean(actual))
        if mean_actual > 0.01:
            denom = np.where(actual == 0, np.nan, actual)
            mape = float(np.nanmean(np.abs(residuals / denom))) * 100
        else:
            mape = 0.0

        metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "training_points": len(historical_df),
        }

        logger.debug(
            f"Model trained: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%"
        )
        return model, metrics

    except Exception as e:
        logger.exception(
            "Prophet model training failed. history_shape=%s params=%s",
            historical_df.shape if isinstance(historical_df, pd.DataFrame) else None,
            params,
        )
        raise RuntimeError(f"Prophet model training failed: {e}") from e


def forecast_demand(
    model: Prophet,
    periods: int = FORECAST_DAYS,
) -> pd.DataFrame:
    """
    Generate future demand predictions using a trained Prophet model.

    Args:
        model: Trained Prophet model.
        periods: Number of future days to forecast.

    Returns:
        DataFrame with forecast columns: ds, yhat, yhat_lower, yhat_upper,
        trend, weekly, etc.
    """
    try:
        safe_periods = max(int(periods), 1)
        future = model.make_future_dataframe(periods=safe_periods)
        forecast = model.predict(future)

        if forecast is None or forecast.empty:
            raise RuntimeError("Prophet returned an empty forecast.")

        if len(future) != len(forecast):
            common_len = min(len(future), len(forecast))
            logger.warning(
                "Prophet output length mismatch: future shape=%s forecast shape=%s; aligning to %s rows",
                future.shape,
                forecast.shape,
                common_len,
            )
            forecast = forecast.iloc[:common_len].copy()

        required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        missing_cols = [col for col in required_cols if col not in forecast.columns]
        if missing_cols:
            raise RuntimeError(f"Prophet forecast missing columns: {missing_cols}")

        forecast = forecast.sort_values("ds").drop_duplicates("ds", keep="last")
        for col in ["yhat", "yhat_lower", "yhat_upper", "trend", "weekly"]:
            if col in forecast.columns:
                forecast[col] = pd.to_numeric(forecast[col], errors="coerce")

        forecast[["yhat", "yhat_lower", "yhat_upper"]] = (
            forecast[["yhat", "yhat_lower", "yhat_upper"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return forecast.reset_index(drop=True)

    except Exception:
        logger.exception(
            "forecast_demand failed at Prophet prediction stage for periods=%s",
            periods,
        )
        raise


def evaluate_model(
    model: Prophet,
    historical_df: pd.DataFrame,
    test_periods: int = 14,
) -> Dict[str, float]:
    """
    Evaluate a trained Prophet model on a holdout test period.

    Splits the last `test_periods` days from the history, forecasts them,
    and computes error metrics against actual.

    Args:
        model: Trained Prophet model.
        historical_df: Full historical DataFrame.
        test_periods: Number of recent days to hold out for testing.

    Returns:
        Dict of evaluation metrics (mae, rmse, mape).
    """
    if len(historical_df) <= test_periods:
        logger.warning(
            f"Not enough data for evaluation "
            f"({len(historical_df)} rows, need > {test_periods})"
        )
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    train_df = historical_df.iloc[:-test_periods]
    test_df = historical_df.iloc[-test_periods:]

    # Retrain on training set
    eval_model, _ = train_model(train_df)

    # Forecast the test period
    future = eval_model.make_future_dataframe(periods=test_periods)
    forecast = eval_model.predict(future)

    # Align forecast with test data
    forecast_dates = forecast[["ds", "yhat"]].tail(test_periods).copy()
    merged = test_df[["ds", "y"]].merge(forecast_dates, on="ds", how="inner")

    if merged.empty:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    residuals = merged["y"].values - merged["yhat"].values
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(residuals / merged["y"].values))) * 100

    logger.debug(
        f"Evaluation on {len(merged)} test points: "
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%"
    )
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "test_points": len(merged),
    }


def _forecast_single_product(
    product_data: Dict[str, Any],
    config: Optional[AppConfig] = None,
) -> ForecastResult:
    """
    Run the full forecasting pipeline for a single product.

    Steps:
    1. Generate historical demand data (synthetic from aggregate stats)
    2. Train Prophet model
    3. Generate forecast
    4. Compute confidence and trend
    5. Return structured ForecastResult

    Args:
        product_data: Dict with product_id, product_name, sales_volume,
                      demand_trend, and optional historical_data.
        config: Application config.

    Returns:
        ForecastResult with predicted demand and metadata.
    """
    pid = str(product_data.get("product_id", ""))
    pname = str(product_data.get("product_name", ""))
    sales_volume = max(_safe_float(product_data.get("sales_volume", 100), 100.0), 0.0)
    demand_trend = np.clip(_safe_float(product_data.get("demand_trend", 0.5), 0.5), 0.0, 1.0)

    result = ForecastResult(
        product_id=pid,
        product_name=pname,
    )

    # Check if historical time-series data is provided
    historical_df = product_data.get("_historical_data", None)
    forecast_df = pd.DataFrame()
    future_forecast = pd.DataFrame()

    try:
        if historical_df is None:
            # Generate synthetic history
            base_sales = max(sales_volume, 10)
            historical_df = _generate_historical_demand(
                base_sales=base_sales,
                demand_trend=demand_trend,
                days=HISTORY_DAYS,
            )
            logger.debug(
                "[%s] Generated synthetic history: rows=%s base=%.0f trend=%.2f",
                pname,
                len(historical_df),
                base_sales,
                demand_trend,
            )

        if not isinstance(historical_df, pd.DataFrame):
            raise ValueError("Historical data must be a pandas DataFrame.")
        if historical_df.empty or not {"ds", "y"}.issubset(historical_df.columns):
            raise ValueError("Historical data must contain non-empty 'ds' and 'y' columns.")

        historical_df = historical_df[["ds", "y"]].copy()
        historical_df["ds"] = pd.to_datetime(historical_df["ds"], errors="coerce")
        historical_df["y"] = pd.to_numeric(historical_df["y"], errors="coerce")
        historical_df = (
            historical_df.dropna(subset=["ds", "y"])
            .sort_values("ds")
            .drop_duplicates("ds", keep="last")
            .reset_index(drop=True)
        )
        historical_df["y"] = historical_df["y"].clip(lower=0)

        if len(historical_df) < 2:
            raise ValueError(f"Insufficient history after cleaning: {len(historical_df)} rows.")

        logger.debug(
            "[%s] Forecast stage=history_cleaned history_shape=%s y_shape=%s",
            pname,
            historical_df.shape,
            historical_df["y"].values.shape,
        )

        # Train model
        model, train_metrics = train_model(historical_df)

        # Forecast
        forecast_df = forecast_demand(model, periods=FORECAST_DAYS)

        # Extract forecast for the future period (last FORECAST_DAYS rows)
        future_rows = min(FORECAST_DAYS, len(forecast_df))
        future_forecast = forecast_df.tail(future_rows).copy()
        if future_forecast.empty:
            raise RuntimeError("No future forecast rows available.")

        # Aggregate predictions
        yhat, lower, upper = _align_arrays(
            future_forecast["yhat"].values,
            future_forecast["yhat_lower"].values,
            future_forecast["yhat_upper"].values,
        )
        if len(yhat) == 0:
            raise RuntimeError("Forecast arrays are empty after alignment.")
        logger.debug(
            "[%s] Forecast stage=future_aligned yhat_shape=%s lower_shape=%s upper_shape=%s",
            pname,
            yhat.shape,
            lower.shape,
            upper.shape,
        )

        total_demand = float(np.maximum(yhat, 0).sum())
        ci_lower = float(np.maximum(lower, 0).sum())
        ci_upper = float(np.maximum(upper, 0).sum())

        # Confidence score
        confidence = _compute_confidence_from_model(model, pd.DataFrame(), forecast_df)

        # Trend detection
        trend = _detect_trend(future_forecast if "trend" in future_forecast.columns else forecast_df)

        # Seasonality strength
        seasonality = _compute_seasonality_strength(model, forecast_df)

        # Weekly breakdown
        weekly = {}
        future_forecast = future_forecast.reset_index(drop=True)
        for week in range(4):
            week_start = week * 7
            week_end = min(week_start + 7, len(future_forecast))
            week_slice = future_forecast.iloc[week_start:week_end]
            week_demand = float(week_slice["yhat"].clip(lower=0).sum()) if not week_slice.empty else 0.0
            weekly[f"week_{week + 1}"] = round(week_demand, 1)

        result.predicted_demand = round(total_demand, 1)
        result.forecast_confidence = round(confidence, 3)
        result.forecast_ci_lower = round(ci_lower, 1)
        result.forecast_ci_upper = round(ci_upper, 1)
        result.demand_trend = trend
        result.weekly_forecast = weekly
        result.seasonality_strength = round(seasonality, 3)

        logger.debug(
            f"[{pname}] Forecast: {total_demand:.0f} units "
            f"(CI: {ci_lower:.0f}-{ci_upper:.0f}, "
            f"conf={confidence:.2f}, trend={trend})"
        )

    except Exception as e:
        logger.exception(
            "Forecast failed for %s (product_id=%s). "
            "history_shape=%s forecast_shape=%s future_shape=%s",
            pname,
            pid,
            getattr(historical_df, "shape", None),
            getattr(forecast_df, "shape", None),
            getattr(future_forecast, "shape", None),
        )
        result = _fallback_forecast_result(
            product_data,
            error_message=str(e),
            confidence=0.50,
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BATCH FORECASTING
# ═══════════════════════════════════════════════════════════════════════════

def batch_forecast(
    df: pd.DataFrame,
    sales_col: str = "sales_volume",
    trend_col: str = "demand_trend",
    max_workers: int = 4,
    save_models: bool = False,
    model_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, ForecastingReport]:
    """
    Run demand forecasting for ALL products in a DataFrame.

    For each product:
    1. Generates synthetic historical demand (or uses provided history)
    2. Trains a Prophet model
    3. Forecasts demand for next 30 days
    4. Computes confidence, trend, weekly breakdown

    Uses concurrent workers for parallel execution.

    Args:
        df: DataFrame with product data. Must contain at minimum
            product_id and sales_volume.
        sales_col: Column with aggregate sales volume.
        trend_col: Column with demand trend signal (0-1).
        max_workers: Max concurrent forecasting threads.
        save_models: If True, saves Prophet models to disk.
        model_dir: Directory to save models (default: models/demand_models/).

    Returns:
        Tuple of:
        - Enriched DataFrame with forecast columns:
            predicted_demand, forecast_confidence,
            forecast_ci_lower, forecast_ci_upper,
            demand_trend_category, forecast_week_1...4
        - ForecastingReport with aggregate statistics.
    """
    if not PROPHET_AVAILABLE:
        logger.warning(
            "Prophet not installed. Install with: pip install prophet cmdstanpy"
        )
        # Fallback mode: use statistical forecast
        return _fallback_batch_forecast(df, sales_col, trend_col)

    if not _prophet_backend_available():
        logger.warning(
            "Prophet is installed but its Stan backend is unavailable. "
            "Reason: %s. Using statistical fallback for the whole batch.",
            _PROPHET_BACKEND_ERROR,
        )
        return _fallback_batch_forecast(df, sales_col, trend_col)

    report = ForecastingReport()
    report.total_products = len(df)

    logger.info(
        f"batch_forecast: {len(df)} products, {max_workers} workers"
    )

    if df.empty:
        result_df = df.copy()
        for col in [
            "predicted_demand", "forecast_confidence", "forecast_ci_lower",
            "forecast_ci_upper", "demand_trend_category", "forecast_next_30d",
            "forecast_week_1", "forecast_week_2", "forecast_week_3", "forecast_week_4",
        ]:
            result_df[col] = []
        return result_df, report

    # Prepare product data dicts
    product_list: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        product_data = {
            "_row_position": position,
            "product_id": str(row.get("product_id", "")),
            "product_name": str(row.get("product_name", "")),
            "sales_volume": max(_safe_float(row.get(sales_col, 100), 100.0), 0.0),
            "demand_trend": np.clip(_safe_float(row.get(trend_col, 0.5), 0.5), 0.0, 1.0),
        }

        # Check for pre-existing historical data
        if "_historical_data" in df.columns:
            hist = row.get("_historical_data")
            if hist is not None and isinstance(hist, pd.DataFrame):
                product_data["_historical_data"] = hist

        product_list.append(product_data)

    # Parallel forecasting
    results: List[Optional[ForecastResult]] = [None] * len(product_list)
    errors: int = 0
    error_ids: List[str] = []

    worker_count = max(1, min(int(max_workers or 1), len(product_list)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        config = AppConfig()
        futures = {
            executor.submit(
                _forecast_single_product, p, config
            ): p
            for p in product_list
        }

        for future in as_completed(futures):
            product = futures[future]
            position = int(product.get("_row_position", 0))
            try:
                result = future.result()
                if result.error_message:
                    errors += 1
                    error_ids.append(result.product_id)
                results[position] = result
            except Exception as e:
                logger.exception(
                    "Forecast failed for %s (product_id=%s) in batch future. "
                    "Returning fallback and continuing batch.",
                    product.get("product_name", ""),
                    product.get("product_id", ""),
                )
                errors += 1
                error_ids.append(str(product.get("product_id", "")))
                results[position] = _fallback_forecast_result(
                    product,
                    error_message=str(e),
                    confidence=0.50,
                )

    final_results: List[ForecastResult] = []
    for product, result in zip(product_list, results):
        if result is None:
            errors += 1
            error_ids.append(str(product.get("product_id", "")))
            result = _fallback_forecast_result(
                product,
                error_message="No forecast result returned.",
                confidence=0.50,
            )
        final_results.append(result)

    report.forecasted = len(final_results) - errors
    report.errors = errors
    report.error_ids = error_ids

    # Build enriched DataFrame
    result_df = df.copy()
    result_df["predicted_demand"] = [r.predicted_demand for r in final_results]
    result_df["forecast_confidence"] = [r.forecast_confidence for r in final_results]
    result_df["forecast_ci_lower"] = [r.forecast_ci_lower for r in final_results]
    result_df["forecast_ci_upper"] = [r.forecast_ci_upper for r in final_results]
    result_df["demand_trend_category"] = [r.demand_trend for r in final_results]

    # Weekly breakdown columns
    for week_num in range(1, 5):
        col_name = f"forecast_week_{week_num}"
        result_df[col_name] = [
            r.weekly_forecast.get(f"week_{week_num}", 0.0) for r in final_results
        ]

    # 30-day total
    result_df["forecast_next_30d"] = result_df["predicted_demand"]

    # Compute aggregate report stats
    valid_results = [
        r for r in final_results
        if r.predicted_demand > 0 or r.forecast_confidence > 0
    ]

    if valid_results:
        report.avg_predicted_demand = float(np.mean([r.predicted_demand for r in valid_results]))
        report.avg_confidence = float(np.mean([r.forecast_confidence for r in valid_results]))
        report.high_confidence_count = sum(
            1 for r in valid_results if r.forecast_confidence >= HIGH_CONFIDENCE_THRESHOLD
        )
        report.medium_confidence_count = sum(
            1 for r in valid_results
            if MEDIUM_CONFIDENCE_THRESHOLD <= r.forecast_confidence < HIGH_CONFIDENCE_THRESHOLD
        )
        report.low_confidence_count = sum(
            1 for r in valid_results if r.forecast_confidence < MEDIUM_CONFIDENCE_THRESHOLD
        )

        for r in valid_results:
            trend_key = r.demand_trend if r.demand_trend in report.trend_distribution else "stable"
            report.trend_distribution[trend_key] += 1

    # Save models if requested
    if save_models:
        _save_forecast_models(final_results, model_dir)

    logger.info(
        f"Batch forecast complete: {report.forecasted} forecasted, "
        f"{report.errors} errors, avg_conf={report.avg_confidence:.2f}"
    )
    return result_df, report


def _fallback_batch_forecast(
    df: pd.DataFrame,
    sales_col: str = "sales_volume",
    trend_col: str = "demand_trend",
) -> Tuple[pd.DataFrame, ForecastingReport]:
    """
    Fallback forecasting when Prophet is not installed.

    Uses simple statistical methods:
    - Predicted demand = sales_volume × demand_trend adjustment
    - Confidence based on data availability

    Args:
        df: Product DataFrame.
        sales_col: Sales volume column.
        trend_col: Demand trend column.

    Returns:
        Tuple of (enriched DataFrame, ForecastingReport).
    """
    logger.warning("Prophet unavailable; using statistical fallback.")

    result_df = df.copy()
    report = ForecastingReport()
    report.total_products = len(df)

    if sales_col in df.columns:
        sales = pd.to_numeric(df[sales_col], errors="coerce").fillna(0).clip(lower=0)
    else:
        sales = pd.Series(0.0, index=df.index)

    if trend_col in df.columns:
        trend = pd.to_numeric(df[trend_col], errors="coerce").fillna(0.5).clip(0, 1)
    else:
        trend = pd.Series(0.5, index=df.index)

    trend_adj = (trend - 0.5) * 0.2 + 1.0
    sales_arr, trend_adj_arr = _align_arrays(sales.values, trend_adj.values)
    result_df["predicted_demand"] = pd.Series(
        sales_arr * trend_adj_arr,
        index=result_df.index[: len(sales_arr)],
    ).reindex(result_df.index, fill_value=0).round(1)
    result_df["forecast_confidence"] = 0.5
    result_df["forecast_ci_lower"] = (result_df["predicted_demand"] * 0.8).round(1)
    result_df["forecast_ci_upper"] = (result_df["predicted_demand"] * 1.2).round(1)
    result_df["demand_trend_category"] = trend.reindex(result_df.index, fill_value=0.5).apply(
        lambda x: "increasing" if x >= 0.6 else ("declining" if x <= 0.4 else "stable")
    )
    result_df["forecast_next_30d"] = result_df["predicted_demand"]

    for week_num in range(1, 5):
        result_df[f"forecast_week_{week_num}"] = (
            result_df["predicted_demand"] / 4
        ).round(1)

    report.forecasted = len(df)
    report.avg_predicted_demand = (
        float(result_df["predicted_demand"].mean()) if len(result_df) else 0.0
    )
    report.avg_confidence = 0.5 if len(result_df) else 0.0

    for trend_cat in ["increasing", "stable", "declining"]:
        report.trend_distribution[trend_cat] = int(
            (result_df["demand_trend_category"] == trend_cat).sum()
        )

    return result_df, report

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _save_forecast_models(
    results: List[ForecastResult],
    model_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Save forecast results as JSON for future reference.

    Note: Prophet models are large to serialize. We save the forecast
    results instead, which contain the actionable data.

    Args:
        results: List of ForecastResult objects.
        model_dir: Directory to save to.
    """
    config = AppConfig()
    save_dir = Path(model_dir) if model_dir else config.get_model_path(MODEL_DIR_NAME)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_dir / f"forecast_results_{timestamp}.pkl"

    data = {
        "generated_at": timestamp,
        "results": [r.to_dict() for r in results],
    }

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Forecast results saved to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save forecast results: {e}")


def load_forecast_results(
    filepath: Union[str, Path],
) -> Optional[List[Dict[str, Any]]]:
    """
    Load previously saved forecast results.

    Args:
        filepath: Path to saved results file.

    Returns:
        List of forecast result dicts, or None if loading fails.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Forecast results not found: {path}")
        return None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Forecast results loaded from {path}")
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Failed to load forecast results: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API
# ═══════════════════════════════════════════════════════════════════════════

class DemandForecaster:
    """
    Enterprise demand forecasting engine using Facebook Prophet.

    Predicts future demand for each product using time-series decomposition
    (trend, weekly seasonality). Supports batch processing for 1000+ products
    with concurrent execution.

    When actual time-series data is not available, the engine generates
    realistic synthetic historical data from aggregate sales_volume and
    demand_trend signals.

    Features:
    - Prophet-based time-series forecasting
    - Synthetic historical data generation
    - Confidence scoring (0-1)
    - Trend detection (increasing / stable / declining)
    - Weekly demand breakdown
    - Model persistence (save/load results)
    - Fallback mode when Prophet is not installed

    Usage:
        forecaster = DemandForecaster()
        df, report = forecaster.batch_forecast(dataframe)
        print(report.summary())
        print(df["predicted_demand"].iloc[0])

        # Single product
        result = forecaster.forecast_single(product_dict)
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        max_workers: int = 4,
        save_models: bool = False,
    ) -> None:
        """
        Initialize the demand forecaster.

        Args:
            config: Application configuration.
            max_workers: Max concurrent forecasting threads.
            save_models: Whether to save forecast results to disk.
        """
        self.config = config or AppConfig()
        self.max_workers = max_workers
        self.save_models = save_models
        self._last_report: Optional[ForecastingReport] = None

    @property
    def last_report(self) -> Optional[ForecastingReport]:
        """Get the report from the most recent batch forecast."""
        return self._last_report

    def forecast_single(
        self,
        product_data: Dict[str, Any],
    ) -> ForecastResult:
        """
        Forecast demand for a single product.

        Args:
            product_data: Dict with product_id, product_name,
                         sales_volume, demand_trend.

        Returns:
            ForecastResult with predicted demand and metadata.
        """
        return _forecast_single_product(product_data, self.config)

    def batch_forecast(
        self,
        df: pd.DataFrame,
        sales_col: str = "sales_volume",
        trend_col: str = "demand_trend",
    ) -> Tuple[pd.DataFrame, ForecastingReport]:
        """
        Forecast demand for all products in the DataFrame.

        Args:
            df: Product DataFrame.
            sales_col: Sales volume column.
            trend_col: Demand trend column.

        Returns:
            Tuple of (enriched DataFrame, ForecastingReport).
        """
        result_df, report = batch_forecast(
            df,
            sales_col=sales_col,
            trend_col=trend_col,
            max_workers=self.max_workers,
            save_models=self.save_models,
        )
        self._last_report = report
        return result_df, report

    def forecast_with_history(
        self,
        product_id: str,
        historical_df: pd.DataFrame,
        periods: int = FORECAST_DAYS,
    ) -> ForecastResult:
        """
        Forecast demand for a product using actual historical time-series data.

        Args:
            product_id: Product identifier.
            historical_df: DataFrame with 'ds' (datetime) and 'y' (sales) columns.
            periods: Number of future days to forecast.

        Returns:
            ForecastResult.
        """
        product_data = {
            "product_id": product_id,
            "product_name": product_id,
            "sales_volume": float(historical_df["y"].mean() * 30),
            "demand_trend": 0.5,
            "_historical_data": historical_df,
        }
        return _forecast_single_product(product_data, self.config)

    def train_and_evaluate(
        self,
        historical_df: pd.DataFrame,
        test_periods: int = 14,
    ) -> Dict[str, Any]:
        """
        Train a Prophet model and evaluate on a holdout set.

        Args:
            historical_df: DataFrame with 'ds' and 'y' columns.
            test_periods: Days to hold out for testing.

        Returns:
            Dict with training metrics and evaluation metrics.
        """
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet not installed"}

        model, train_metrics = train_model(historical_df)
        eval_metrics = evaluate_model(model, historical_df, test_periods=test_periods)

        return {
            "training": train_metrics,
            "evaluation": eval_metrics,
        }

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last batch forecast."""
        if self._last_report is None:
            return "No batch forecast performed yet."
        return self._last_report.summary()
