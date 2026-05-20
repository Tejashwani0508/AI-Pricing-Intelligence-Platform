"""
Application Configuration Module

Centralized configuration management for the AI Pricing Intelligence Platform.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

@dataclass
class AppConfig:
    """
    Application configuration settings.

    All configurable parameters for risk thresholds, alerts, forecasting,
    and file paths are centralized here for easy maintenance.
    """

    # ─── Paths ───────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    reports_dir: Path = base_dir / "reports"

    # ─── Risk Thresholds ─────────────────────────────────────────────
    margin_erosion_threshold: float = 0.15       # 15% margin erosion flags risk
    price_volatility_threshold: float = 0.20      # 20% price variance flags volatility
    markdown_risk_threshold: float = 0.25         # 25% below competitor flags markdown
    high_risk_score: float = 70.0                 # Score above this = high risk
    medium_risk_score: float = 40.0               # Score above this = medium risk

    # ─── Alert Thresholds ────────────────────────────────────────────
    price_change_threshold: float = 0.10          # 10% price change triggers alert
    price_above_competitor_threshold: float = 0.15  # 15% above competitor triggers alert
    inventory_low_threshold: int = 30             # Days of cover below this = low stock
    inventory_excess_threshold: int = 180         # Days of cover above this = excess stock
    sales_decline_threshold: float = -0.20        # 20% sales drop triggers alert

    # ─── Forecasting ─────────────────────────────────────────────────
    forecast_periods: int = 30                    # Number of future periods to forecast
    confidence_level: float = 0.95                # Confidence interval for forecasts
    seasonality_period: int = 7                   # Days for seasonality detection

    # ─── Pricing Engine ──────────────────────────────────────────────
    min_price_multiplier: float = 0.7             # Floor: 70% of current price
    max_price_multiplier: float = 1.5             # Ceiling: 150% of current price
    target_margin: float = 0.40                   # Desired 40% margin
    min_margin: float = 0.10                      # Minimum acceptable 10% margin

    # ─── Data Processing ─────────────────────────────────────────────
    batch_size: int = 100                         # Products per batch for concurrent processing
    max_workers: int = 4                          # Parallel worker threads
    required_columns: List[str] = field(
        default_factory=lambda: [
            "product_id", "product_name", "category", "current_price",
            "cost_price", "competitor_price", "sales_volume",
            "inventory_level", "demand_trend", "price_elasticity",
        ]
    )

    # ─── Database (optional) ─────────────────────────────────────────
    db_host: Optional[str] = os.getenv("DB_HOST", None)
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: Optional[str] = os.getenv("DB_NAME", None)
    db_user: Optional[str] = os.getenv("DB_USER", None)
    db_password: Optional[str] = os.getenv("DB_PASSWORD", None)

    # ─── API Keys (future use) ───────────────────────────────────────
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    @property
    def margin_risk_columns(self) -> List[str]:
        """Columns used for margin risk assessment."""
        return ["current_price", "cost_price", "competitor_price"]

    @property
    def demand_risk_columns(self) -> List[str]:
        """Columns used for demand risk assessment."""
        return ["sales_volume", "demand_trend", "price_elasticity"]

    @property
    def inventory_risk_columns(self) -> List[str]:
        """Columns used for inventory risk assessment."""
        return ["inventory_level", "sales_volume"]

    def get_report_path(self, filename: str) -> Path:
        """Get full path for a report file."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return self.reports_dir / filename

    def get_model_path(self, filename: str) -> Path:
        """Get full path for a model file."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir / filename