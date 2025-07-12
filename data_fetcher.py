"""
Market data fetching utilities.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def fetch_spot_price(ticker: str) -> float:
    """Fetch current spot price for a given ticker."""
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        return float(data["Close"].dropna().iloc[-1])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch price for {ticker}: {e}")


def time_to_expiration(expiry_str: str) -> float:
    """Calculate time to expiration in years."""
    expiry = pd.to_datetime(expiry_str)
    delta = (expiry - datetime.now()).total_seconds()
    return max(delta / (365.25 * 24 * 3600), 0)


def fetch_option_chain(ticker: str, spot_price: float) -> pd.DataFrame:
    """Fetch and filter option chain data."""
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options

    if not expiration_dates:
        raise ValueError("No option expiration data available")

    # Combine all expiration dates
    calls_frames = []
    for date in expiration_dates:
        try:
            df = stock.option_chain(date).calls
            df['expiration'] = date
            calls_frames.append(df)
        except Exception:
            continue

    if not calls_frames:
        raise ValueError("No valid option data found")

    calls_all = pd.concat(calls_frames, ignore_index=True)

    # Filter by strike range and time
    strike_range = (spot_price * 0.7, spot_price * 1.3)
    mask = (calls_all['strike'] >= strike_range[0]) & (calls_all['strike'] <= strike_range[1])
    calls_filtered = calls_all[mask].copy()

    calls_filtered['TimeToExpiry'] = calls_filtered['expiration'].apply(time_to_expiration)
    calls_filtered = calls_filtered[calls_filtered['TimeToExpiry'] >= 0.07]

    return calls_filtered