"""
Data models for Black-Scholes option pricing.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholes:
    """Black-Scholes option pricing model."""

    T: float  # Time to expiration
    K: float  # Strike price
    S: float  # Current stock price
    sigma: float  # Volatility
    r: float  # Risk-free rate

    def _d1_d2(self) -> tuple[float, float]:
        """Calculate d1 and d2 parameters."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def prices(self) -> tuple[float, float]:
        """Calculate call and put option prices."""
        d1, d2 = self._d1_d2()
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put


@dataclass
class OptionData:
    """Container for option market data."""

    ticker: str
    spot_price: float
    strike: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    plot_by: str = 'strike'