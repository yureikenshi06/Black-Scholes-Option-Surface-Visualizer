from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp

@dataclass
class BlackScholes:
    """Black-Scholes option pricing model."""
    T: float  # Time to expiration
    K: float  # Strike price
    S: float  # Spot price
    sigma: float  # Volatility
    r: float  # Risk-free rate
    q: float = 0.0  # Dividend yield

    def _d1_d2(self):
        """Calculate d1 and d2 parameters."""
        d1 = (log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        return d1, d2

    def prices(self):
        """Calculate call and put option prices."""
        d1, d2 = self._d1_d2()
        call = exp(-self.q * self.T) * self.S * norm.cdf(d1) - exp(-self.r * self.T) * self.K * norm.cdf(d2)
        put = exp(-self.r * self.T) * self.K * norm.cdf(-d2) - exp(-self.q * self.T) * self.S * norm.cdf(-d1)
        return call, put