"""
Implied volatility calculations.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from typing import Optional


def bs_call_price(S: float, K: float, r: float, T: float, sigma: float, q: float = 0.0) -> float:
    """Calculate Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return (np.exp(-q * T) * S * norm.cdf(d1) -
            np.exp(-r * T) * K * norm.cdf(d2))


def implied_volatility(S: float, K: float, r: float, T: float,
                       market_price: float, q: float = 0.0) -> Optional[float]:
    """Calculate implied volatility using Newton's method."""
    if market_price < max(0, S - K * np.exp(-r * T)):
        return None

    try:
        objective = lambda sigma: bs_call_price(S, K, r, T, sigma, q) - market_price
        return newton(objective, 0.2, tol=1e-5, maxiter=100)
    except Exception:
        return None


def calculate_implied_volatilities(option_data, spot_price: float, risk_free_rate: float):
    """Calculate implied volatilities for option chain data."""
    ivs = []

    for _, row in option_data.iterrows():
        if row['TimeToExpiry'] > 0:
            iv = implied_volatility(
                spot_price,
                row['strike'],
                risk_free_rate,
                row['TimeToExpiry'],
                row['lastPrice']
            )
            ivs.append(iv)
        else:
            ivs.append(None)

    option_data['ImpliedVolatility'] = ivs
    return option_data.dropna(subset=['ImpliedVolatility'])