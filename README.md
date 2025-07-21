# Black-Scholes Options Surface Visualizer

A Python application for visualizing implied volatility and option price surfaces using the Black-Scholes model.

## Features

- **Real-time Data Fetching**: Retrieves live stock prices and option chain data from Yahoo Finance
- **Implied Volatility Calculation**: Computes implied volatility using Newton-Raphson method
- **3D Surface Visualization**: Interactive 3D plots for volatility and price surfaces
- **Flexible Analysis**: Option to plot by strike price or moneyness
- **Professional Architecture**: Clean, modular codebase with separation of concerns

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download the repository
2. Navigate to the project directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python main.py
```

### Interactive Prompts
The application will prompt you for:
- **Stock Ticker**: Enter symbol (e.g., AAPL, MSFT, GOOGL)
- **Risk-free Rate**: Enter as decimal (e.g., 0.05 for 5%)
- **Dividend Yield**: Enter as decimal (e.g., 0.02 for 2%)
- **Plot Type**: Choose between "Strike Price" or "Moneyness"

### Example Session
```
Enter stock ticker (e.g. AAPL): AAPL
Enter risk-free rate (e.g. 0.05): 0.05
Enter dividend yield (e.g. 0.00): 0.015
Plot by [Strike Price / Moneyness] (default = Strike Price): moneyness
```

## Project Structure

```
├── main.py              # Main application entry point
├── models.py            # BlackScholes data model
├── data_fetcher.py      # Yahoo Finance data retrieval
├── option_calculator.py # Option pricing calculations
├── surface_plotter.py   # 3D visualization utilities
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Module Overview

### `models.py`
- `BlackScholes`: Core option pricing model with call/put price calculations

### `data_fetcher.py`
- `DataFetcher`: Handles live market data retrieval and filtering

### `option_calculator.py`
- `OptionCalculator`: Implied volatility calculations and surface generation

### `surface_plotter.py`
- `SurfacePlotter`: Interactive 3D plotting with Plotly

### `main.py`
- User interface and application orchestration

## Output

The application generates three interactive 3D surface plots:
1. **Implied Volatility Surface**: Shows how IV varies with time and strike/moneyness
2. **Call Price Surface**: Theoretical call option prices using Black-Scholes
3. **Put Price Surface**: Theoretical put option prices using Black-Scholes

