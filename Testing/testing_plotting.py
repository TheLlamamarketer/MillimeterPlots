import numpy as np
from plotting_old import plot_data

def generate_data(n_points, x_min, x_max, slope, intercept, noise_level):
    """Generate synthetic linear data with noise."""
    xdata = np.linspace(x_min, x_max, n_points)
    ydata = slope * xdata + np.random.normal(0, noise_level, n_points) + intercept
    return xdata, ydata

# Generate multiple datasets
xdata1, ydata1 = generate_data(50, 0, 100, 1, 20, 5)
xdata2, ydata2 = generate_data(50, 0, 100, 0.3, 10, 3)
xdata3, ydata3 = generate_data(50, 0, 100, 0.5, 80, 2)

# Error bars
y_error1 = [5] * len(ydata1)
y_error2 = [3] * len(ydata2)
y_error3 = [2] * len(ydata3)

# Prepare datasets for plotting
datasets = [
    {'xdata': xdata1, 'ydata': ydata1, 'y_error': y_error1, 'label': 'Dataset 1'},
    {'xdata': xdata2, 'ydata': ydata2, 'y_error': y_error2, 'label': 'Dataset 2'},
    {'xdata': xdata3, 'ydata': ydata3, 'y_error': y_error3, 'label': 'Dataset 3'}
]

# Define grey zones to highlight specific areas
grey_zones = [{'x_low': 20, 'x_up': 30}, {'y_low': 70, 'y_up': 90}, {'x_val': 50}]

# Call plot_data with enhanced options and multiple datasets
plot_data(
    filename='Plots/test.pdf', datasets=datasets, ymin=0, ymax=160, 
    ystep=10, xmin=0, xmax=100, xstep=10, grey_zones=grey_zones, 
    background_color='#eb3107', x_label='X-axis', y_label='Y-axis', legend_position='upper left'
)
