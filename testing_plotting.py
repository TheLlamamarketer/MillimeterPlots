import numpy as np
from plotting import plot_data

def generate_data(n_points, x_min, x_max, slope, intercept, noise_level):
    xdata = np.linspace(x_min, x_max, n_points)
    ydata = slope * xdata + np.random.normal(0, noise_level, n_points) + intercept
    return xdata, ydata

xdata, ydata = generate_data(10, 0, 100, 1, 20, 3)

data = {
    'x':xdata.tolist(),
    'y':ydata.tolist()
}

def error(data, percent, digit):
    return [i * percent + digit for i in data]


# each {} is a zone with conditions, meaning each zone could be only 1 point or a range of points, variable in both x and y or both
grey_zones = [
    {'x_low': 10, 'x_up': 20},  
    {'y_low': 70, 'y_up': 100},  
    {'x_val': 120}, 
]

# Example usage
plot_data(ymin=0, ymax=140, xmin=0, xmax=100, xstep=10, ystep=10, xtickoffset= 0, ydata=data['y'], xdata=data['x'], y_error= error(data['y'], 0, 10), grey_zones=grey_zones, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/test.pdf')
