import numpy as np
from data_fetcher import DataFetcher
from option_calculator import OptionCalculator
from surface_plotter import SurfacePlotter


def get_user_inputs():
    """Get user inputs for the analysis."""
    print("\n" + "=" * 75)
    print("           📈 Black-Scholes IV & Price Surface Visualizer")
    print("=" * 75 + "\n")

    ticker = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol is required.")

    r = float(input("Enter risk-free rate (e.g. 0.05): ") or 0.05)
    q = float(input("Enter dividend yield (e.g. 0.00): ") or 0.00)

    plot_choice = input("Plot by [Strike Price / Moneyness] (default = Strike Price): ").strip().lower()
    use_moneyness = plot_choice in ['moneyness', 'm']

    return ticker, r, q, use_moneyness


def process_option_data(calls_data, S, r, q, use_moneyness):
    """Process option data and calculate implied volatilities."""
    # Calculate implied volatilities
    ivs = []
    for _, row in calls_data.iterrows():
        if row['TimeToExpiry'] > 0:
            iv = OptionCalculator.implied_volatility(
                S=S, K=row['strike'], r=r, T=row['TimeToExpiry'],
                call_price=row['lastPrice'], q=q
            )
            ivs.append(iv)
        else:
            ivs.append(np.nan)

    calls_data['ImpliedVolatility'] = ivs
    calls_data = calls_data.dropna(subset=['ImpliedVolatility'])

    if calls_data.empty:
        raise ValueError("No valid implied volatility data to plot.")

    # Setup plot variables
    if use_moneyness:
        calls_data['Moneyness'] = calls_data['strike'] / S
        Y = calls_data['Moneyness'].values
        y_label = 'Moneyness'
        y_format = '.3f'
    else:
        Y = calls_data['strike'].values
        y_label = 'Strike Price ($)'
        y_format = '.2f'

    X = calls_data['TimeToExpiry'].values
    Z_iv = calls_data['ImpliedVolatility'].values * 100

    return X, Y, Z_iv, y_label, y_format


def main():
    """Main application entry point."""
    try:
        # Get user inputs
        ticker, r, q, use_moneyness = get_user_inputs()

        # Fetch market data
        print(f"\n[INFO] Fetching data for {ticker}...")
        S = DataFetcher.fetch_spot(ticker)
        print(f"[INFO] Latest spot price: ${S:.2f}")

        calls_data = DataFetcher.fetch_option_chain(ticker, S)
        print(f"[INFO] Processing {len(calls_data)} option contracts...")

        # Process option data
        X, Y, Z_iv, y_label, y_format = process_option_data(calls_data, S, r, q, use_moneyness)

        # Create interpolated surfaces
        plotter = SurfacePlotter()
        xi, yi, zi_iv = plotter.interpolate_surface(X, Y, Z_iv)

        # Calculate option price surfaces
        zi_call, zi_put = OptionCalculator.calculate_option_surfaces(xi, yi, S, r, q, use_moneyness)

        # Create and display plots
        print("[INFO] Generating surface plots...")

        iv_fig = plotter.create_surface(xi, yi, zi_iv, f"Implied Volatility Surface - {ticker}",
                                        y_label, "Volatility (%)", y_format)
        call_fig = plotter.create_surface(xi, yi, zi_call, f"Call Price Surface - {ticker}",
                                          y_label, "Call Price ($)", y_format)
        put_fig = plotter.create_surface(xi, yi, zi_put, f"Put Price Surface - {ticker}",
                                         y_label, "Put Price ($)", y_format)

        iv_fig.show()
        call_fig.show()
        put_fig.show()

        print("[INFO] Analysis complete!")

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()