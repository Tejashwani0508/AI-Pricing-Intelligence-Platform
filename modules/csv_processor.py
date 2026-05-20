"""
CSV / Data Processor Module

Robust, production-grade CSV processing system for the AI Pricing Intelligence Platform.
Handles upload, validation, cleaning, type conversion, duplicate removal, and error reporting
for datasets with 1000+ products.

Provides both standalone functions (functional API) and the CSVProcessor class (OOP API).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.config import AppConfig
from utils.helpers import validate_numeric_column

logger = logging.getLogger("ai_pricing.csv_processor")


# ─── Error Types ─────────────────────────────────────────────────────────

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class ColumnMappingError(DataValidationError):
    """Raised when required columns cannot be mapped from the source."""
    pass


class TypeConversionError(DataValidationError):
    """Raised when type coercion fails irrecoverably."""
    pass


# ─── Column Mapping ──────────────────────────────────────────────────────

# Default column mapping: standardised_name -> list of accepted source names.
# This allows users to upload files with varying column naming conventions.
DEFAULT_COLUMN_MAP: Dict[str, List[str]] = {
    "product_id": ["product_id", "productid", "id", "sku", "code"],
    "product_name": [
        "product_name", "productname", "product", "name", "item", "title",
    ],
    "category": ["category", "cat", "product_category", "department", "group"],
    "current_price": [
        "current_price", "price", "selling_price", "retail_price",
        "sale_price", "unit_price", "list_price",
    ],
    "cost_price": [
        "cost_price", "cost", "unit_cost", "cogs", "purchase_price",
        "wholesale_price", "supplier_price",
    ],
    "competitor_price": [
        "competitor_price", "competitorprice", "competitor_pricing",
        "market_price", "comp_price", "avg_competitor_price",
    ],
    "sales_volume": [
        "sales_volume", "sales", "volume", "units_sold", "quantity_sold",
        "demand_quantity", "demand",
    ],
    "inventory_level": [
        "inventory_level", "inventory", "stock", "stock_level",
        "on_hand", "available_qty", "qty_on_hand",
    ],
    "demand_trend": [
        "demand_trend", "demandscore", "demand_score", "trend",
        "demand_signal", "popularity", "popularity_score",
    ],
    "price_elasticity": [
        "price_elasticity", "elasticity", "price_sensitivity",
        "elasticity_coefficient", "demand_elasticity",
    ],
}

# Minimum accepted column set for processing (must map at least these).
MINIMUM_REQUIRED_COLUMNS: List[str] = [
    "product_id",
    "current_price",
    "cost_price",
]

# Numeric columns and their expected ranges.
NUMERIC_COLUMNS: Dict[str, Dict[str, Optional[float]]] = {
    "current_price": {"min": 0.01, "max": None},
    "cost_price": {"min": 0.01, "max": None},
    "competitor_price": {"min": 0.01, "max": None},
    "sales_volume": {"min": 0, "max": None},
    "inventory_level": {"min": 0, "max": None},
    "demand_trend": {"min": 0.0, "max": 1.0},
    "price_elasticity": {"min": -10.0, "max": 0.0},
}


# ─── Report Data Class ───────────────────────────────────────────────────

@dataclass
class ProcessingReport:
    """
    Detailed report of the CSV processing pipeline.

    Captures all actions taken, warnings raised, errors encountered,
    and transformation steps so callers can inspect exactly what happened.
    """
    total_rows_input: int = 0
    total_rows_output: int = 0
    columns_found: List[str] = field(default_factory=list)
    columns_mapped: Dict[str, str] = field(default_factory=dict)
    columns_missing: List[str] = field(default_factory=list)
    columns_ignored: List[str] = field(default_factory=list)

    duplicates_removed: int = 0
    rows_with_missing_values: int = 0
    rows_with_out_of_range: int = 0
    rows_dropped_total: int = 0

    type_conversion_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    median_imputations: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a plain dictionary for serialisation."""
        return {
            "total_rows_input": self.total_rows_input,
            "total_rows_output": self.total_rows_output,
            "rows_dropped": self.rows_dropped_total,
            "duplicates_removed": self.duplicates_removed,
            "rows_with_missing_values": self.rows_with_missing_values,
            "rows_with_out_of_range": self.rows_with_out_of_range,
            "columns_found": self.columns_found,
            "columns_mapped": self.columns_mapped,
            "columns_missing": self.columns_missing,
            "columns_ignored": self.columns_ignored,
            "type_conversion_errors": len(self.type_conversion_errors),
            "validation_errors": self.validation_errors,
            "warnings": self.warnings,
            "median_imputations": self.median_imputations,
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"  Rows read:         {self.total_rows_input}",
            f"  Rows output:       {self.total_rows_output}",
            f"  Duplicates removed: {self.duplicates_removed}",
            f"  Missing-value rows: {self.rows_with_missing_values}",
            f"  Out-of-range rows:  {self.rows_with_out_of_range}",
            f"  Total dropped:     {self.rows_dropped_total}",
            f"  Columns mapped:    {len(self.columns_mapped)}",
            f"  Columns missing:   {len(self.columns_missing)}",
            f"  Warnings:          {len(self.warnings)}",
        ]
        if self.validation_errors:
            lines.append(f"  Validation errors: {len(self.validation_errors)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTIONS  (Functional API)
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    low_memory: bool = False,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, ProcessingReport]:
    """
    Load a CSV file into a pandas DataFrame with basic diagnostics.

    Args:
        file_path: Path to the CSV file.
        encoding: File encoding (default utf-8).
        low_memory: Passed to pd.read_csv; False ensures consistent dtypes.
        **kwargs: Additional keyword arguments forwarded to pd.read_csv.

    Returns:
        Tuple of (raw DataFrame, ProcessingReport with load diagnostics).

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the file cannot be parsed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    report = ProcessingReport()

    # 1. Quick file sanity check ---------------------------------------
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Loading CSV: {path} ({file_size_mb:.2f} MB)")

    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=low_memory,
                **kwargs,
            )
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"CSV file is empty: {path}")
        except pd.errors.ParserError as e:
            raise DataValidationError(
                f"CSV parsing error in {path}: {e}. "
                "Check for inconsistent delimiters or quoting."
            ) from e
        except UnicodeDecodeError as e:
            raise DataValidationError(
                f"Encoding error in {path}: {e}. "
                f"Try a different encoding (e.g., latin1, cp1252)."
            ) from e
    elif path.suffix.lower() in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(path, engine="openpyxl", **kwargs)
        except Exception as e:
            raise DataValidationError(f"Excel read error: {e}") from e
    else:
        raise ValueError(
            f"Unsupported format: {path.suffix}. Use .csv, .xls, or .xlsx."
        )

    report.total_rows_input = len(df)
    report.columns_found = list(df.columns)

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    logger.info(
        f"Loaded {len(df)} rows, {len(df.columns)} columns "
        f"(columns: {list(df.columns)})"
    )
    return df, report


def validate_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, List[str]]] = None,
    minimum_required: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, ProcessingReport]:
    """
    Validate and map source columns to standardised column names.

    The function tries to find each standard column in the source
    DataFrame by matching against a list of accepted aliases.

    Args:
        df: Raw DataFrame from load_csv().
        column_map: Dict mapping standard name -> list of source aliases.
                     Defaults to DEFAULT_COLUMN_MAP.
        minimum_required: List of standard columns that must be present
                          after mapping. Defaults to MINIMUM_REQUIRED_COLUMNS.

    Returns:
        Tuple of (DataFrame with renamed columns, ProcessingReport).

    Raises:
        ColumnMappingError: If any minimum-required column cannot be mapped.
    """
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP
    if minimum_required is None:
        minimum_required = MINIMUM_REQUIRED_COLUMNS

    report = ProcessingReport()
    source_cols_lower = set(c.lower() for c in df.columns)
    rename_dict: Dict[str, str] = {}
    missing: List[str] = []
    ignored: List[str] = []

    # Try to map each standard column
    for standard_name, aliases in column_map.items():
        found = False
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in source_cols_lower:
                # Find the actual column name (preserve original casing)
                actual = next(c for c in df.columns if c.lower() == alias_lower)
                rename_dict[actual] = standard_name
                found = True
                break
        if not found:
            missing.append(standard_name)

    # Check minimum required columns
    still_missing = [c for c in minimum_required if c in missing]
    if still_missing:
        raise ColumnMappingError(
            f"Required columns could not be mapped: {still_missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected aliases: {[column_map[c] for c in still_missing]}"
        )

    # Apply renaming
    df_renamed = df.rename(columns=rename_dict)

    # Track ignored (unmapped) columns
    for col in df.columns:
        if col not in rename_dict:
            ignored.append(col)

    report.columns_mapped = {v: k for k, v in rename_dict.items()}
    report.columns_missing = missing
    report.columns_ignored = ignored

    if ignored:
        logger.info(f"Ignored unmapped columns: {ignored}")

    return df_renamed, report


def clean_data(
    df: pd.DataFrame,
    numeric_columns: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Tuple[pd.DataFrame, ProcessingReport]:
    """
    Clean the DataFrame by coercing types, handling invalid values,
    and removing duplicates.

    Steps:
    1. Strip whitespace from string columns.
    2. Coerce numeric columns to proper types (errors -> NaN).
    3. Remove duplicate rows.
    4. Validate numeric ranges.

    Args:
        df: DataFrame with mapped column names (from validate_columns).
        numeric_columns: Dict of standard_name -> {min, max} ranges.
                         Defaults to NUMERIC_COLUMNS.

    Returns:
        Tuple of (cleaned DataFrame, ProcessingReport).
    """
    if numeric_columns is None:
        numeric_columns = NUMERIC_COLUMNS

    report = ProcessingReport()
    df_clean = df.copy()
    total_before = len(df_clean)

    # 1. Strip whitespace from string columns ---------------------------
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Coerce numeric columns -----------------------------------------
    for col, bounds in numeric_columns.items():
        if col not in df_clean.columns:
            continue
        orig_vals = df_clean[col].copy()
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Count coercion failures
        coerced_mask = orig_vals.notna() & df_clean[col].isna()
        failed_count = coerced_mask.sum()
        if failed_count > 0:
            report.type_conversion_errors.append(
                {
                    "column": col,
                    "count": int(failed_count),
                    "examples": orig_vals[coerced_mask].head(3).tolist(),
                }
            )
            logger.warning(
                f"Column '{col}': {failed_count} values could not be "
                f"converted to numeric and were set to NaN. "
                f"Examples: {orig_vals[coerced_mask].head(3).tolist()}"
            )

    # 3. Remove duplicates ----------------------------------------------
    id_cols = ["product_id", "product_name"]
    id_col_present = [c for c in id_cols if c in df_clean.columns]

    if id_col_present:
        dupe_before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=id_col_present, keep="first")
        report.duplicates_removed = dupe_before - len(df_clean)
        if report.duplicates_removed > 0:
            logger.info(
                f"Removed {report.duplicates_removed} duplicate rows "
                f"based on {id_col_present}."
            )
    else:
        # Fallback: drop fully duplicated rows
        dupe_before = len(df_clean)
        df_clean = df_clean.drop_duplicates(keep="first")
        report.duplicates_removed = dupe_before - len(df_clean)

    # 4. Validate numeric ranges ----------------------------------------
    rows_before_range = len(df_clean)
    for col, bounds in numeric_columns.items():
        if col not in df_clean.columns:
            continue
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        df_clean = validate_numeric_column(
            df_clean, col, min_val=min_val, max_val=max_val
        )
    report.rows_with_out_of_range = rows_before_range - len(df_clean)
    report.rows_dropped_total = total_before - len(df_clean)

    logger.info(
        f"Cleaning complete: {total_before} -> {len(df_clean)} rows "
        f"({report.rows_dropped_total} dropped)"
    )
    return df_clean, report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    numeric_columns: Optional[List[str]] = None,
    fill_value: Optional[float] = None,
) -> Tuple[pd.DataFrame, ProcessingReport]:
    """
    Handle missing values in the DataFrame.

    Strategies:
    - "median" : Fill numeric columns with the column median.
    - "mean"   : Fill with the column mean.
    - "zero"   : Fill with 0.
    - "drop"   : Drop rows containing any NaN in numeric columns.
    - "ffill"  : Forward-fill.
    - "bfill"  : Backward-fill.
    - "value"  : Fill with the explicit `fill_value` argument.

    Args:
        df: Cleaned DataFrame.
        strategy: Imputation strategy (default "median").
        numeric_columns: List of columns to impute. If None, uses all
                         present NUMERIC_COLUMNS keys.
        fill_value: Explicit value used when strategy="value".

    Returns:
        Tuple of (DataFrame with missing values handled, ProcessingReport).
    """
    if numeric_columns is None:
        numeric_columns = [c for c in NUMERIC_COLUMNS if c in df.columns]

    report = ProcessingReport()
    df_out = df.copy()
    total_before = len(df_out)

    null_mask = df_out[numeric_columns].isna()
    report.rows_with_missing_values = int(null_mask.any(axis=1).sum())

    if report.rows_with_missing_values == 0:
        logger.info("No missing values found; skipping imputation.")
        return df_out, report

    logger.info(
        f"Handling missing values: {report.rows_with_missing_values} rows affected "
        f"using strategy='{strategy}'"
    )

    if strategy == "drop":
        df_out = df_out.dropna(subset=numeric_columns)
        report.rows_dropped_total = total_before - len(df_out)

    elif strategy == "median":
        for col in numeric_columns:
            null_count = int(null_mask[col].sum())
            if null_count > 0:
                median_val = df_out[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                    report.warnings.append(
                        f"Column '{col}' median is NaN; using 0.0 as fallback."
                    )
                df_out[col] = df_out[col].fillna(median_val)
                report.median_imputations[col] = null_count
                logger.debug(f"  {col}: {null_count} NaN values → median={median_val:.4f}")

    elif strategy == "mean":
        for col in numeric_columns:
            null_count = int(null_mask[col].sum())
            if null_count > 0:
                mean_val = df_out[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                df_out[col] = df_out[col].fillna(mean_val)

    elif strategy == "zero":
        for col in numeric_columns:
            df_out[col] = df_out[col].fillna(0.0)

    elif strategy == "ffill":
        df_out[numeric_columns] = df_out[numeric_columns].ffill()

    elif strategy == "bfill":
        df_out[numeric_columns] = df_out[numeric_columns].bfill()

    elif strategy == "value":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='value'.")
        for col in numeric_columns:
            df_out[col] = df_out[col].fillna(fill_value)
    else:
        raise ValueError(
            f"Unknown imputation strategy: '{strategy}'. "
            f"Valid: median, mean, zero, drop, ffill, bfill, value."
        )

    # Final check: report any remaining NaN
    remaining_nulls = df_out[numeric_columns].isna().sum().sum()
    if remaining_nulls > 0:
        report.warnings.append(
            f"{remaining_nulls} NaN values remain after '{strategy}' imputation."
        )

    logger.info(
        f"Missing values handled: {report.rows_with_missing_values} rows affected, "
        f"{remaining_nulls} remaining NaN"
    )
    return df_out, report


def process_dataset(
    file_path: Union[str, Path],
    column_map: Optional[Dict[str, List[str]]] = None,
    minimum_required: Optional[List[str]] = None,
    numeric_columns: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    missing_strategy: str = "median",
    fill_value: Optional[float] = None,
    engineer_features: bool = True,
    **load_kwargs: Any,
) -> Tuple[pd.DataFrame, ProcessingReport]:
    """
    End-to-end CSV processing pipeline.

    Combines load_csv → validate_columns → clean_data → handle_missing_values
    into a single call. Optionally runs feature engineering.

    This is the primary entry point for most use cases.

    Args:
        file_path: Path to the CSV/Excel file.
        column_map: Column alias mapping (defaults to DEFAULT_COLUMN_MAP).
        minimum_required: Must-have columns (defaults to MINIMUM_REQUIRED).
        numeric_columns: Numeric column range specs.
        missing_strategy: Imputation strategy for missing values.
        fill_value: Explicit fill value (used when strategy="value").
        engineer_features: Whether to run feature engineering after cleaning.
        **load_kwargs: Additional args passed to load_csv / pd.read_csv.

    Returns:
        Tuple of (cleaned, processed DataFrame, ProcessingReport).

    Raises:
        FileNotFoundError, DataValidationError, ColumnMappingError.
    """
    logger.info(f"=== process_dataset start: {file_path} ===")

    # Step 1: Load
    df, report_load = load_csv(file_path, **load_kwargs)

    # Step 2: Validate & map columns
    df_mapped, report_cols = validate_columns(
        df, column_map=column_map, minimum_required=minimum_required,
    )

    # Step 3: Clean data (type coercion, duplicates, range validation)
    df_clean, report_clean = clean_data(
        df_mapped, numeric_columns=numeric_columns,
    )

    # Step 4: Handle missing values
    df_final, report_missing = handle_missing_values(
        df_clean,
        strategy=missing_strategy,
        numeric_columns=list(
            (numeric_columns or NUMERIC_COLUMNS).keys()
        ) if numeric_columns else None,
        fill_value=fill_value,
    )

    # Merge reports
    report_final = report_missing
    report_final.total_rows_input = report_load.total_rows_input
    report_final.total_rows_output = len(df_final)
    report_final.columns_found = report_load.columns_found
    report_final.columns_mapped = report_cols.columns_mapped
    report_final.columns_missing = report_cols.columns_missing
    report_final.columns_ignored = report_cols.columns_ignored
    report_final.duplicates_removed = report_clean.duplicates_removed
    report_final.rows_with_out_of_range = report_clean.rows_with_out_of_range
    report_final.type_conversion_errors = report_clean.type_conversion_errors
    report_final.validation_errors = report_clean.validation_errors + report_cols.validation_errors
    report_final.rows_dropped_total = (
        report_load.total_rows_input - len(df_final)
    )

    logger.info(
        f"=== process_dataset complete: "
        f"{report_final.total_rows_input} → {report_final.total_rows_output} rows ==="
    )
    return df_final, report_final


def generate_error_report(report: ProcessingReport) -> str:
    """
    Generate a detailed human-readable error report string.

    Args:
        report: ProcessingReport from any processing step.

    Returns:
        Formatted error/warning report string.
    """
    lines: List[str] = [
        "=" * 60,
        "  CSV PROCESSING REPORT",
        "=" * 60,
        "",
    ]
    lines.append(report.summary())
    lines.append("")

    if report.type_conversion_errors:
        lines.append("── Type Conversion Errors ──")
        for err in report.type_conversion_errors:
            lines.append(
                f"  Column '{err['column']}': {err.get('count', '?')} values "
                f"failed. Examples: {err.get('examples', [])}"
            )
        lines.append("")

    if report.validation_errors:
        lines.append("── Validation Errors ──")
        for ve in report.validation_errors:
            lines.append(f"  • {ve}")
        lines.append("")

    if report.warnings:
        lines.append("── Warnings ──")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    if report.columns_ignored:
        lines.append(f"Ignored (unmapped) columns: {report.columns_ignored}")

    if report.columns_missing:
        lines.append(
            f"Optional columns not found: {report.columns_missing}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API  (backward-compatible wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class CSVProcessor:
    """
    Processes product pricing data from CSV or Excel files.

    Wraps the standalone functional API in a class for backward compatibility.
    Supports both the original exact-column-schema and flexible column mapping.

    Handles:
    - File loading (CSV, Excel) with flexible column mapping
    - Column validation and mapping from aliases
    - Data cleaning and type coercion
    - Duplicate removal
    - Missing value handling (median imputation)
    - Feature engineering (margin, price gaps, etc.)

    Usage:
        processor = CSVProcessor()
        df = processor.load_file("products.csv")
        df = processor.engineer_features()
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        column_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Initialize the CSV processor.

        Args:
            config: Application configuration (uses defaults if None).
            column_map: Custom column alias mapping (uses DEFAULT_COLUMN_MAP if None).
        """
        self.config = config or AppConfig()
        self.column_map = column_map or DEFAULT_COLUMN_MAP
        self.required_columns: List[str] = self.config.required_columns
        self._df: Optional[pd.DataFrame] = None
        self._report: Optional[ProcessingReport] = None

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the currently loaded DataFrame."""
        return self._df

    @property
    def report(self) -> Optional[ProcessingReport]:
        """Get the last processing report."""
        return self._report

    def load_file(
        self,
        file_path: Union[str, Path],
        missing_strategy: str = "median",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from a CSV or Excel file with full processing pipeline.

        Args:
            file_path: Path to the input file.
            missing_strategy: Imputation strategy for missing values.
            **kwargs: Additional arguments for load_csv / pd.read_csv.

        Returns:
            Cleaned and processed DataFrame.

        Raises:
            FileNotFoundError: If file doesn't exist.
            DataValidationError: If validation fails.
            ValueError: If file format is unsupported.
        """
        df, report = process_dataset(
            file_path,
            column_map=self.column_map,
            missing_strategy=missing_strategy,
            engineer_features=False,
            **kwargs,
        )
        self._df = df
        self._report = report

        logger.info(
            f"File loaded: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def load_dataframe(
        self,
        df: pd.DataFrame,
        missing_strategy: str = "median",
    ) -> pd.DataFrame:
        """
        Load data from an existing DataFrame (in-memory source).

        Cleans and validates the DataFrame using the processing pipeline.

        Args:
            df: Input DataFrame.
            missing_strategy: Imputation strategy for missing values.

        Returns:
            Cleaned and validated DataFrame.
        """
        logger.info(f"Loading DataFrame with {len(df)} rows")

        # Map columns using the alias mapping
        df_mapped, col_report = validate_columns(
            df, column_map=self.column_map
        )

        # Clean
        df_clean, clean_report = clean_data(df_mapped)

        # Handle missing
        df_final, miss_report = handle_missing_values(
            df_clean, strategy=missing_strategy
        )

        self._df = df_final
        self._report = miss_report

        logger.info(
            f"DataFrame processed: {len(df_final)} rows "
            f"({len(df) - len(df_final)} dropped)"
        )
        return df_final

    def engineer_features(self) -> pd.DataFrame:
        """
        Create derived features for analysis.

        Calculates:
        - profit_margin: (price - cost) / price
        - price_vs_competitor: price / competitor_price
        - inventory_turnover: sales_volume / inventory_level
        - revenue: price * sales_volume
        - days_of_cover: inventory / daily_sales
        - demand_score: demand_trend * elasticity_adjustment

        Returns:
            DataFrame with additional feature columns.
        """
        if self._df is None or self._df.empty:
            raise DataValidationError(
                "No data loaded. Call load_file() or load_dataframe() first."
            )

        df = self._df.copy()

        # Margin calculations
        if "current_price" in df.columns and "cost_price" in df.columns:
            df["profit_margin"] = (
                (df["current_price"] - df["cost_price"]) / df["current_price"]
            ).clip(lower=0.0)
        else:
            logger.warning("Cannot compute profit_margin: missing price/cost columns.")

        # Competitor comparisons
        if "current_price" in df.columns and "competitor_price" in df.columns:
            df["price_vs_competitor"] = (
                df["current_price"]
                / df["competitor_price"].replace(0, float("nan"))
            ).fillna(1.0)
        else:
            logger.warning(
                "Cannot compute competitor features: missing competitor_price column."
            )

        # Inventory metrics
        if "sales_volume" in df.columns and "inventory_level" in df.columns:
            df["inventory_turnover"] = (
                df["sales_volume"]
                / df["inventory_level"].replace(0, float("nan"))
            ).fillna(0.0)

            df["days_of_cover"] = (
                (
                    df["inventory_level"]
                    / df["sales_volume"].replace(0, float("nan"))
                )
                * 30
            ).fillna(999.0).clip(upper=999.0)
        else:
            logger.warning(
                "Cannot compute inventory metrics: missing sales_volume "
                "or inventory_level columns."
            )

        # Revenue
        if "current_price" in df.columns and "sales_volume" in df.columns:
            df["revenue"] = df["current_price"] * df["sales_volume"]

        # Demand-weighted score
        if (
            "demand_trend" in df.columns
            and "price_elasticity" in df.columns
        ):
            df["demand_score"] = (
                df["demand_trend"]
                * (1 - abs(df["price_elasticity"]) * 0.1)
            ).clip(0.0, 1.0)
        elif "demand_trend" in df.columns:
            df["demand_score"] = df["demand_trend"]

        self._df = df
        logger.info(
            f"Feature engineering complete: {len(df.columns)} columns"
        )
        return df

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for numerical columns.

        Returns:
            Dictionary with column stats (mean, median, min, max, std).
        """
        if self._df is None or self._df.empty:
            return {}

        numeric_df = self._df.select_dtypes(include="number")
        stats: Dict[str, Dict[str, float]] = {}

        for col in numeric_df.columns:
            stats[col] = {
                "mean": round(float(numeric_df[col].mean()), 2),
                "median": round(float(numeric_df[col].median()), 2),
                "min": round(float(numeric_df[col].min()), 2),
                "max": round(float(numeric_df[col].max()), 2),
                "std": round(float(numeric_df[col].std()), 2),
            }

        return stats

    def get_category_summary(self) -> pd.DataFrame:
        """
        Get aggregated metrics by product category.

        Returns:
            DataFrame grouped by category with aggregate stats.
        """
        if self._df is None or self._df.empty:
            return pd.DataFrame()

        if "category" not in self._df.columns:
            logger.warning("Cannot compute category summary: no 'category' column.")
            return pd.DataFrame()

        cat_cols = {
            "current_price": "mean",
            "cost_price": "mean",
            "competitor_price": "mean",
            "profit_margin": "mean",
            "sales_volume": "sum",
            "inventory_level": "sum",
            "revenue": "sum",
            "product_id": "count",
        }

        existing_cols = {
            c: agg
            for c, agg in cat_cols.items()
            if c in self._df.columns
        }

        summary = self._df.groupby("category").agg(existing_cols)
        summary.columns = [
            f"avg_{c}" if agg == "mean" else f"total_{c}"
            for c, agg in existing_cols.items()
        ]
        summary = summary.rename(columns={"total_product_id": "product_count"})
        summary = summary.round(2).reset_index()

        return summary

    def get_processing_report(self) -> Optional[Dict[str, Any]]:
        """
        Get the detailed processing report as a dictionary.

        Returns:
            Report dict or None if no processing has been done.
        """
        if self._report is None:
            return None
        return self._report.to_dict()

    def validate_raw_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, ProcessingReport]:
        """
        Validate and clean a raw DataFrame without loading it as the
        processor's internal state. Useful for previewing or multi-step flows.

        Args:
            df: Raw input DataFrame.

        Returns:
            Tuple of (cleaned DataFrame, ProcessingReport).
        """
        df_mapped, _ = validate_columns(df, column_map=self.column_map)
        df_clean, clean_report = clean_data(df_mapped)
        df_final, miss_report = handle_missing_values(df_clean)

        # Merge reports
        miss_report.columns_mapped = clean_report.columns_mapped
        miss_report.columns_missing = clean_report.columns_missing
        miss_report.columns_ignored = clean_report.columns_ignored
        miss_report.duplicates_removed = clean_report.duplicates_removed
        miss_report.type_conversion_errors = clean_report.type_conversion_errors

        return df_final, miss_report
