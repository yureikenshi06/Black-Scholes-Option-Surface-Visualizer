"""
User interface utilities for input handling.
"""
from typing import Optional, List


def prompt_float(prompt_text: str, default: Optional[float] = None,
                 min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Prompt user for float input with validation."""
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
            if (min_val is not None and fval < min_val) or \
                    (max_val is not None and fval > max_val):
                print(f"Value must be between {min_val} and {max_val}. Please try again.")
                continue
            return fval
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def prompt_choice(prompt_text: str, choices: List[str],
                  default: Optional[str] = None) -> str:
    """Prompt user for choice from a list of options."""
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

        print(f"Invalid choice. Please enter one of: {choices_str}")


def display_summary(option_data) -> None:
    """Display input summary."""
    print("\n=== Input Summary ===")
    print(f"Ticker:           {option_data.ticker}")
    print(f"Spot Price:       {option_data.spot_price:.2f}")
    print(f"Strike Price:     {option_data.strike:.2f}")
    print(f"Time to Maturity: {option_data.time_to_expiry:.4f} years")
    print(f"Volatility:       {option_data.volatility:.4f}")
    print(f"Risk-free Rate:   {option_data.risk_free_rate:.4f}")
    print(f"Plot by:          {option_data.plot_by}\n")


def display_prices(call_price: float, put_price: float) -> None:
    """Display calculated option prices."""
    print(f"Calculated Call Price: {call_price:,.2f}")
    print(f"Calculated Put Price:  {put_price:,.2f}\n")