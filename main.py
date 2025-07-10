
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from numpy import log, sqrt, exp
from scipy.optimize import newton
from scipy.interpolate import griddata
import plotly.graph_objects as go
import argparse
from datetime import datetime

@dataclass
class BlackScholes:
    T: float
    K: float
    S: float
    sigma: float
    r: float

    def _d1_d2(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / \
             (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        return d1, d2

    def prices(self):
        d1, d2 = self._d1_d2()
        call = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put

def fetch_spot(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df.empty:
            raise ValueError("No data received for the ticker.")
        return float(df["Close"].dropna().iloc[-1])
    except Exception as e:
        sys.exit(f"Failed to fetch price for {ticker}: {e}")

def time_to_expiration(expiry_str):
    expiry = pd.to_datetime(expiry_str)
    now = datetime.now()
    delta = (expiry - now).total_seconds()
    return max(delta / (365.25 * 24 * 3600), 0)

def bs_call_price(S, X, r, T, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(0, S - X)
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * X * norm.cdf(d2)

def call_implied_vol(S, X, r, T, call_price, q=0.0):
    if call_price < max(0, S - X * np.exp(-r * T)):
        return np.nan
    try:
        func = lambda sigma: bs_call_price(S, X, r, T, sigma, q) - call_price
        return newton(func, 0.2, tol=1e-5, maxiter=100)
    except Exception:
        return np.nan

def plot_surface(xi, yi, zi, customdata, hovertemplate, title, y_label):
    fig = go.Figure(data=[go.Surface(
        x=xi, y=yi, z=zi,
        colorscale='Viridis',
        customdata=customdata,
        hovertemplate=hovertemplate
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time to Expiration (years)',
            yaxis_title=y_label,
            zaxis_title=title.split()[0]
        ),
        width=800,
        height=700
    )
    fig.show()

def prompt_float(prompt_text, default=None, min_val=None, max_val=None):
    while True:
        prompt = f"{prompt_text}"
        if default is not None:
            prompt += f" [default: {default}]: "
        else:
            prompt += ": "
        val = input(prompt).strip()
        if val == "" and default is not None:
            return default
        try:
            fval = float(val)
            if (min_val is not None and fval < min_val) or (max_val is not None and fval > max_val):
                print(f"Value must be between {min_val} and {max_val}. Please try again.")
                continue
            return fval
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def prompt_choice(prompt_text, choices, default=None):
    choices_str = "/".join(choices)
    while True:
        prompt = f"{prompt_text} ({choices_str})"
        if default is not None:
            prompt += f" [default: {default}]: "
        else:
            prompt += ": "
        val = input(prompt).strip().lower()
        if val == "" and default is not None:
            return default.lower()
        if val in choices:
            return val
        else:
            print(f"Invalid choice. Please enter one of: {choices_str}")

def main():
    parser = argparse.ArgumentParser(description="Interactive Black-Scholes Option Pricer")
    parser.add_argument('--ticker', type=str, help='Ticker symbol, e.g. AAPL')
    parser.add_argument('--strike', type=float, help='Strike price')
    parser.add_argument('--maturity', type=float, help='Time to maturity in years')
    parser.add_argument('--vol', type=float, help='Volatility (e.g. 0.2)')
    parser.add_argument('--rate', type=float, help='Risk-free interest rate')
    parser.add_argument('--plot-by', type=str, choices=['strike', 'moneyness'], help='Plot IV surface by strike or moneyness')

    # Fix for Jupyter/Colab argv injection:
    import sys
    if 'ipykernel' in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    # Interactive prompts only for missing inputs
    ticker = args.ticker or input("Enter ticker symbol (e.g. AAPL): ").strip().upper()
    spot = fetch_spot(ticker)
    print(f"Latest spot price for {ticker}: {spot:.2f}")

    strike = args.strike
    if strike is None:
        strike = prompt_float("Enter strike price", default=spot, min_val=0)

    maturity = args.maturity
    if maturity is None:
        maturity = prompt_float("Enter time to maturity (years)", default=1.0, min_val=0)

    vol = args.vol
    if vol is None:
        vol = prompt_float("Enter volatility (e.g. 0.2)", default=0.2, min_val=0)

    rate = args.rate
    if rate is None:
        rate = prompt_float("Enter risk-free rate (e.g. 0.05)", default=0.05)

    plot_by = args.plot_by
    if plot_by is None:
        plot_by = prompt_choice("Plot IV surface by 'strike' or 'moneyness'", ['strike', 'moneyness'], default='strike')

    print("\n=== Input Summary ===")
    print(f"Ticker:           {ticker}")
    print(f"Spot Price:       {spot:.2f}")
    print(f"Strike Price:     {strike:.2f}")
    print(f"Time to Maturity: {maturity:.4f} years")
    print(f"Volatility:       {vol:.4f}")
    print(f"Risk-free Rate:   {rate:.4f}")
    print(f"Plot by:          {plot_by}\n")

    bs = BlackScholes(maturity, strike, spot, vol, rate)
    call_price, put_price = bs.prices()

    print(f"Calculated Call Price: {call_price:,.2f}")
    print(f"Calculated Put Price:  {put_price:,.2f}\n")

    # Fetch option chain for IV surface
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    if not expiration_dates:
        sys.exit("No option expiration data available.")

    calls_frames = []
    for date in expiration_dates:
        df = stock.option_chain(date).calls
        df['expiration'] = date
        calls_frames.append(df)
    calls_all = pd.concat(calls_frames, ignore_index=True)

    min_strike = spot * 0.7
    max_strike = spot * 1.3
    mask = (calls_all['strike'] >= min_strike) & (calls_all['strike'] <= max_strike)
    calls_filtered = calls_all[mask].copy()
    calls_filtered['TimeToExpiry'] = calls_filtered['expiration'].apply(time_to_expiration)
    calls_filtered = calls_filtered[calls_filtered['TimeToExpiry'] >= 0.07]

    ivs = []
    for _, row in calls_filtered.iterrows():
        T_ = row['TimeToExpiry']
        if T_ > 0:
            iv = call_implied_vol(spot, row['strike'], rate, T_, row['lastPrice'], 0.0)
            ivs.append(iv)
        else:
            ivs.append(np.nan)
    calls_filtered['ImpliedVolatility'] = ivs
    calls_filtered = calls_filtered.dropna(subset=['ImpliedVolatility'])

    if calls_filtered.empty:
        sys.exit("No valid implied volatility data to plot.")

    if plot_by == 'moneyness':
        calls_filtered['Moneyness'] = calls_filtered['strike'] / spot
        X = calls_filtered['TimeToExpiry'].values
        Y = calls_filtered['Moneyness'].values
        y_label = 'Moneyness'
        y_format_specifier = '.3f'
    else:
        X = calls_filtered['TimeToExpiry'].values
        Y = calls_filtered['strike'].values
        y_label = 'Strike Price ($)'
        y_format_specifier = '.2f'

    Z_iv = calls_filtered['ImpliedVolatility'].values * 100

    xi = np.linspace(min(X), max(X), 30)
    yi = np.linspace(min(Y), max(Y), 30)
    xi, yi = np.meshgrid(xi, yi)

    zi_iv = griddata((X, Y), Z_iv, (xi, yi), method='linear')

    zi_call = np.zeros_like(zi_iv)
    zi_put = np.zeros_like(zi_iv)

    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            T_ = xi[i, j]
            strike_val = yi[i, j] * spot if plot_by == 'moneyness' else yi[i, j]
            if T_ > 0 and strike_val > 0:
                bs_tmp = BlackScholes(T_, strike_val, spot, vol, rate)
                c, p = bs_tmp.prices()
                zi_call[i, j] = c
                zi_put[i, j] = p
            else:
                zi_call[i, j] = np.nan
                zi_put[i, j] = np.nan

    def make_customdata(z, x, y):
        return np.stack((z, x, y), axis=-1)

    hovertemplate_iv = (
        "Volatility: %{customdata[0]:.2f}%<br>"
        "Time to Expiration: %{customdata[1]:.3f} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format_specifier}}}<extra></extra>"
    )
    hovertemplate_call = (
        "Call Price: %{customdata[0]:.2f}<br>"
        "Time to Expiration: %{customdata[1]:.3f} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format_specifier}}}<extra></extra>"
    )
    hovertemplate_put = (
        "Put Price: %{customdata[0]:.2f}<br>"
        "Time to Expiration: %{customdata[1]:.3f} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format_specifier}}}<extra></extra>"
    )

    plot_surface(
        xi, yi, zi_iv,
        make_customdata(zi_iv, xi, yi),
        hovertemplate_iv,
        f"Implied Volatility Surface of {ticker}",
        y_label
    )
    plot_surface(
        xi, yi, zi_call,
        make_customdata(zi_call, xi, yi),
        hovertemplate_call,
        f"Call Price Surface of {ticker}",
        y_label
    )
    plot_surface(
        xi, yi, zi_put,
        make_customdata(zi_put, xi, yi),
        hovertemplate_put,
        f"Put Price Surface of {ticker}",
        y_label
    )

if __name__ == "__main__":
    main()
