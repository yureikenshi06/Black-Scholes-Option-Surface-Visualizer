#!/usr/bin/env python3
"""
Professional Black-Scholes Option Pricing Tool
"""
import sys
import argparse
from models import BlackScholes, OptionData
from data_fetcher import fetch_spot_price, fetch_option_chain
from volatility import calculate_implied_volatilities
from visualization import plot_option_surfaces
from interface import prompt_float, prompt_choice, display_summary, display_prices


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Black-Scholes Option Pricer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--strike', type=float, help='Strike price')
    parser.add_argument('--maturity', type=float, help='Time to maturity in years')
    parser.add_argument('--vol', type=float, help='Volatility (e.g., 0.2 for 20%%)')
    parser.add_argument('--rate', type=float, help='Risk-free interest rate')
    parser.add_argument('--plot-by', type=str, choices=['strike', 'moneyness'],
                        help='Plot IV surface by strike or moneyness')

    # Handle Jupyter/Colab environment
    if 'ipykernel' in sys.modules:
        return parser.parse_args(args=[])

    return parser.parse_args()


def get_user_inputs(args) -> OptionData:
    """Collect user inputs interactively."""
    # Get ticker and fetch spot price
    ticker = args.ticker or input("Enter ticker symbol (e.g., AAPL): ").strip().upper()

    try:
        spot_price = fetch_spot_price(ticker)
        print(f"Latest spot price for {ticker}: {spot_price:.2f}")
    except Exception as e:
        sys.exit(f"Error: {e}")

    # Get other parameters
    strike = args.strike or prompt_float("Enter strike price", default=spot_price, min_val=0)
    maturity = args.maturity or prompt_float("Enter time to maturity (years)", default=1.0, min_val=0)
    volatility = args.vol or prompt_float("Enter volatility (e.g., 0.2)", default=0.2, min_val=0)
    risk_free_rate = args.rate or prompt_float("Enter risk-free rate (e.g., 0.05)", default=0.05)
    plot_by = args.plot_by or prompt_choice(
        "Plot IV surface by 'strike' or 'moneyness'",
        ['strike', 'moneyness'],
        default='strike'
    )

    return OptionData(
        ticker=ticker,
        spot_price=spot_price,
        strike=strike,
        time_to_expiry=maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        plot_by=plot_by
    )


def main():
    """Main application entry point."""
    try:
        args = parse_arguments()
        option_data = get_user_inputs(args)

        display_summary(option_data)

        # Calculate Black-Scholes prices
        bs = BlackScholes(
            option_data.time_to_expiry,
            option_data.strike,
            option_data.spot_price,
            option_data.volatility,
            option_data.risk_free_rate
        )

        call_price, put_price = bs.prices()
        display_prices(call_price, put_price)

        # Fetch and process option chain data
        print("Fetching option chain data...")
        option_chain = fetch_option_chain(option_data.ticker, option_data.spot_price)

        print("Calculating implied volatilities...")
        option_chain_with_iv = calculate_implied_volatilities(
            option_chain,
            option_data.spot_price,
            option_data.risk_free_rate
        )

        if option_chain_with_iv.empty:
            sys.exit("No valid implied volatility data available for plotting.")

        # Generate and display plots
        print("Generating 3D surface plots...")
        plots = plot_option_surfaces(
            option_chain_with_iv,
            option_data.spot_price,
            option_data.volatility,
            option_data.risk_free_rate,
            option_data.ticker,
            option_data.plot_by
        )

        # Show plots
        for plot in plots:
            plot.show()

        print("Analysis complete!")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()