import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataFetcher:
    """Handles fetching market data from Yahoo Finance."""
    
    @staticmethod
    def fetch_spot(ticker: str) -> float:
        """Fetch current spot price for ticker."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if df.empty:
                raise ValueError("No data received for the ticker.")
            return float(df["Close"].dropna().iloc[-1])
        except Exception as e:
            raise Exception(f"Failed to fetch spot price: {e}")
    
    @staticmethod
    def fetch_option_chain(ticker: str, spot_price: float, min_ratio: float = 0.7, max_ratio: float = 1.3):
        """Fetch filtered option chain data."""
        stock = yf.Ticker(ticker)
        expiration_dates = stock.options
        
        if not expiration_dates:
            raise ValueError("No option expiration data available.")
        
        calls_frames = []
        for date in expiration_dates:
            df = stock.option_chain(date).calls
            df['expiration'] = date
            calls_frames.append(df)
        
        calls_all = pd.concat(calls_frames, ignore_index=True)
        
        # Filter by strike range
        min_strike = spot_price * min_ratio
        max_strike = spot_price * max_ratio
        calls_filtered = calls_all[
            (calls_all['strike'] >= min_strike) & 
            (calls_all['strike'] <= max_strike)
        ].copy()
        
        # Add time to expiration
        calls_filtered['TimeToExpiry'] = calls_filtered['expiration'].apply(
            lambda x: max((pd.to_datetime(x) - datetime.now()).total_seconds() / (365.25 * 24 * 3600), 0)
        )
        
        # Filter by minimum time to expiration
        return calls_filtered[calls_filtered['TimeToExpiry'] >= 0.07]