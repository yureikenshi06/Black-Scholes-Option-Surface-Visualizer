"""
3D surface plotting for option data.
"""
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from models import BlackScholes


def create_surface_plot(xi, yi, zi, title: str, y_label: str,
                        customdata=None, hovertemplate=None):
    """Create a 3D surface plot."""
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

    return fig


def generate_surface_data(option_data, spot_price: float, volatility: float,
                          risk_free_rate: float, plot_by: str = 'strike'):
    """Generate interpolated surface data for plotting."""
    if plot_by == 'moneyness':
        option_data['Moneyness'] = option_data['strike'] / spot_price
        X = option_data['TimeToExpiry'].values
        Y = option_data['Moneyness'].values
        y_label = 'Moneyness'
        y_format = '.3f'
    else:
        X = option_data['TimeToExpiry'].values
        Y = option_data['strike'].values
        y_label = 'Strike Price ($)'
        y_format = '.2f'

    Z_iv = option_data['ImpliedVolatility'].values * 100

    # Create meshgrid for interpolation
    xi = np.linspace(min(X), max(X), 30)
    yi = np.linspace(min(Y), max(Y), 30)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate implied volatility
    zi_iv = griddata((X, Y), Z_iv, (xi, yi), method='linear')

    # Calculate theoretical prices
    zi_call = np.zeros_like(zi_iv)
    zi_put = np.zeros_like(zi_iv)

    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            T = xi[i, j]
            strike_val = yi[i, j] * spot_price if plot_by == 'moneyness' else yi[i, j]

            if T > 0 and strike_val > 0:
                bs = BlackScholes(T, strike_val, spot_price, volatility, risk_free_rate)
                call_price, put_price = bs.prices()
                zi_call[i, j] = call_price
                zi_put[i, j] = put_price
            else:
                zi_call[i, j] = np.nan
                zi_put[i, j] = np.nan

    return xi, yi, zi_iv, zi_call, zi_put, y_label, y_format


def plot_option_surfaces(option_data, spot_price: float, volatility: float,
                         risk_free_rate: float, ticker: str, plot_by: str = 'strike'):
    """Plot implied volatility and price surfaces."""
    xi, yi, zi_iv, zi_call, zi_put, y_label, y_format = generate_surface_data(
        option_data, spot_price, volatility, risk_free_rate, plot_by
    )

    def make_customdata(z, x, y):
        return np.stack((z, x, y), axis=-1)

    # Create hover templates
    hovertemplate_iv = (
        f"Volatility: %{{customdata[0]:.2f}}%<br>"
        f"Time to Expiration: %{{customdata[1]:.3f}} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format}}}<extra></extra>"
    )
    hovertemplate_call = (
        f"Call Price: %{{customdata[0]:.2f}}<br>"
        f"Time to Expiration: %{{customdata[1]:.3f}} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format}}}<extra></extra>"
    )
    hovertemplate_put = (
        f"Put Price: %{{customdata[0]:.2f}}<br>"
        f"Time to Expiration: %{{customdata[1]:.3f}} years<br>"
        f"{y_label}: %{{customdata[2]:{y_format}}}<extra></extra>"
    )

    # Create plots
    plots = []

    # Implied Volatility Surface
    iv_fig = create_surface_plot(
        xi, yi, zi_iv,
        f"Implied Volatility Surface of {ticker}",
        y_label,
        make_customdata(zi_iv, xi, yi),
        hovertemplate_iv
    )
    plots.append(iv_fig)

    # Call Price Surface
    call_fig = create_surface_plot(
        xi, yi, zi_call,
        f"Call Price Surface of {ticker}",
        y_label,
        make_customdata(zi_call, xi, yi),
        hovertemplate_call
    )
    plots.append(call_fig)

    # Put Price Surface
    put_fig = create_surface_plot(
        xi, yi, zi_put,
        f"Put Price Surface of {ticker}",
        y_label,
        make_customdata(zi_put, xi, yi),
        hovertemplate_put
    )
    plots.append(put_fig)

    return plots