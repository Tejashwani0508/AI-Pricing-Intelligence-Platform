"""
Helper Functions Module

Shared utility functions used across the AI Pricing Intelligence Platform.
"""

from typing import Any, List, Optional, Union
import pandas as pd
import numpy as np


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a numeric value as a currency string.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string (e.g. "$1,234.56")
    """
    if pd.isna(value) or value is None:
        return "$0.00"
    return f"${value:,.{decimals}f}"


def safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning a default on division by zero.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value returned if denominator is zero

    Returns:
        Division result or default
    """
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return default
    if numerator is None or pd.isna(numerator):
        return default
    return numerator / denominator


def calculate_percentage_change(
    old_value: float, new_value: float
) -> float:
    """
    Calculate the percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change (e.g. 0.1 for 10% increase)
    """
    if old_value is None or old_value == 0 or pd.isna(old_value):
        return 0.0
    if new_value is None or pd.isna(new_value):
        return 0.0
    return (new_value - old_value) / abs(old_value)


def round_half_up(value: float, decimals: int = 2) -> float:
    """
    Round a number using "round half up" method (away from zero).

    Python's built-in round() uses banker's rounding. This provides
    more predictable rounding for pricing calculations.

    Args:
        value: Number to round
        decimals: Number of decimal places

    Returns:
        Rounded value
    """
    multiplier = 10 ** decimals
    return float(
        np.floor(value * multiplier + 0.5) / multiplier
    )


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into smaller chunks of a specified size.

    Useful for batch processing large product datasets.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        return [items]
    return [
        items[i : i + chunk_size]
        for i in range(0, len(items), chunk_size)
    ]


def validate_numeric_column(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> pd.DataFrame:
    """
    Validate and clean a numeric column in a DataFrame.

    Removes rows where the column is NaN or outside the specified range.

    Args:
        df: Input DataFrame
        column: Column name to validate
        min_val: Minimum acceptable value (inclusive)
        max_val: Maximum acceptable value (inclusive)

    Returns:
        DataFrame with invalid rows removed
    """
    mask = df[column].notna()

    if min_val is not None:
        mask &= df[column] >= min_val
    if max_val is not None:
        mask &= df[column] <= max_val

    cleaned = df[mask].copy()
    return cleaned


def categorize_margin(margin: float) -> str:
    """
    Categorize a profit margin percentage.

    Args:
        margin: Margin as a decimal (e.g. 0.35 for 35%)

    Returns:
        Category string: 'High', 'Medium', 'Low', or 'Negative'
    """
    if margin >= 0.40:
        return "High"
    elif margin >= 0.20:
        return "Medium"
    elif margin >= 0.0:
        return "Low"
    return "Negative"


def categorize_risk(score: float) -> str:
    """
    Categorize a risk score into severity levels.

    Args:
        score: Risk score (0-100)

    Returns:
        Risk category: 'Critical', 'High', 'Medium', 'Low'
    """
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


def create_price_bins(
    prices: pd.Series, num_bins: int = 10
) -> pd.Series:
    """
    Create evenly-spaced price bins for histogram/dashboard display.

    Args:
        prices: Series of price values
        num_bins: Number of bins to create

    Returns:
        Series with bin labels
    """
    if prices.empty or prices.nunique() < 2:
        return pd.Series(["N/A"] * len(prices), index=prices.index)

    bins = np.linspace(prices.min(), prices.max(), num_bins + 1)
    labels = [
        f"${bins[i]:.0f}-${bins[i+1]:.0f}"
        for i in range(len(bins) - 1)
    ]
    return pd.cut(prices, bins=bins, labels=labels, include_lowest=True)