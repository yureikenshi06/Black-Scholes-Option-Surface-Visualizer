# Black-Scholes Option Pricing Tool

A professional, modular implementation of the Black-Scholes option pricing model with 3D visualization capabilities.

## Features

- **Black-Scholes Pricing**: Calculate theoretical option prices
- **Implied Volatility**: Extract implied volatility from market data
- **3D Visualization**: Interactive surface plots for volatility and price surfaces
- **Real-time Data**: Fetch live market data via yfinance
- **Flexible Interface**: Command-line arguments or interactive prompts

## Project Structure

```
├── models.py           # Core data models (BlackScholes, OptionData)
├── data_fetcher.py     # Market data fetching utilities
├── volatility.py       # Implied volatility calculations
├── visualization.py    # 3D surface plotting
├── interface.py        # User interface utilities
├── main.py            # Main application entry point
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Interactive mode
python main.py

# With parameters
python main.py --ticker AAPL --strike 150 --maturity 0.25 --vol 0.2 --rate 0.05 --plot-by strike
```

### Parameters

- `--ticker`: Stock ticker symbol (e.g., AAPL, GOOGL)
- `--strike`: Strike price
- `--maturity`: Time to maturity in years
- `--vol`: Volatility (e.g., 0.2 for 20%)
- `--rate`: Risk-free interest rate
- `--plot-by`: Plot by 'strike' or 'moneyness'

## Examples

### Basic Usage

```python
from models import BlackScholes

# Create Black-Scholes model
bs = BlackScholes(T=0.25, K=100, S=105, sigma=0.2, r=0.05)
call_price, put_price = bs.prices()

print(f"Call Price: ${call_price:.2f}")
print(f"Put Price: ${put_price:.2f}")
```

### Implied Volatility Calculation

```python
from volatility import implied_volatility

iv = implied_volatility(
    S=105,          # Spot price
    K=100,          # Strike price
    r=0.05,         # Risk-free rate
    T=0.25,         # Time to expiration
    market_price=8.5 # Market price of option
)

print(f"Implied Volatility: {iv:.2%}")
```

