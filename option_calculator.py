from scipy.optimize import newton


try:
    from models import BlackScholes
except ImportError:
    # Fallback if models.py is not available
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
            d1 = (log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (
                        self.sigma * sqrt(self.T))
            d2 = d1 - self.sigma * sqrt(self.T)
            return d1, d2

        def prices(self):
            """Calculate call and put option prices."""
            d1, d2 = self._d1_d2()
            call = exp(-self.q * self.T) * self.S * norm.cdf(d1) - exp(-self.r * self.T) * self.K * norm.cdf(d2)
            put = exp(-self.r * self.T) * self.K * norm.cdf(-d2) - exp(-self.q * self.T) * self.S * norm.cdf(-d1)
            return call, put


class OptionCalculator:

    @staticmethod
    def bs_call_price(S, K, r, T, sigma, q=0.0):
        """Calculate Black-Scholes call price."""
        if T <= 0 or sigma <= 0:
            return max(0, S - K)

        d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return exp(-q * T) * S * norm.cdf(d1) - exp(-r * T) * K * norm.cdf(d2)

    @staticmethod
    def implied_volatility(S, K, r, T, call_price, q=0.0):
        """Calculate implied volatility using Newton-Raphson method."""
        if call_price < max(0, S - K * exp(-r * T)):
            return np.nan

        try:
            func = lambda sigma: OptionCalculator.bs_call_price(S, K, r, T, sigma, q) - call_price
            return newton(func, 0.2, tol=1e-5, maxiter=100)
        except:
            return np.nan

    @staticmethod
    def calculate_option_surfaces(xi, yi, S, r, q, is_moneyness=False):
        zi_call, zi_put = np.zeros_like(xi), np.zeros_like(xi)

        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                T = xi[i, j]
                K = yi[i, j] * S if is_moneyness else yi[i, j]

                if T > 0 and K > 0:
                    bs = BlackScholes(T, K, S, sigma=0.2, r=r, q=q)
                    call, put = bs.prices()
                    zi_call[i, j] = call
                    zi_put[i, j] = put
                else:
                    zi_call[i, j] = np.nan
                    zi_put[i, j] = np.nan

        return zi_call, zi_put