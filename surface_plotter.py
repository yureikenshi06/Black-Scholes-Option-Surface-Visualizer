import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

class SurfacePlotter:
    @staticmethod
    def create_surface(xi, yi, zi, title, y_label, z_label, y_format='.2f'):
        """Create a 3D surface plot."""
        customdata = np.stack((zi, xi, yi), axis=-1)
        
        hovertemplate = (
            f"{z_label}: %{{customdata[0]:.2f}}<br>" +
            "Time to Expiration: %{customdata[1]:.3f} years<br>" +
            f"{y_label}: %{{customdata[2]:{y_format}}}<extra></extra>"
        )
        
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
                zaxis_title=z_label
            ),
            width=800,
            height=700
        )
        
        return fig
    
    @staticmethod
    def interpolate_surface(X, Y, Z, grid_size=30):
        """Interpolate data onto a regular grid."""
        xi = np.linspace(min(X), max(X), grid_size)
        yi = np.linspace(min(Y), max(Y), grid_size)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((X, Y), Z, (xi, yi), method='linear')
        return xi, yi, zi