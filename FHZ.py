import numpy as np
from collections import defaultdict
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PPoly

data = {}
def process_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    columns = list(zip(*[line.split() for line in lines[2:]]))
    dataset = np.array(columns, dtype=float)
    data[file] = dataset

files = [
    'FHZ_data/FHZ Neon Palex.txt',
    'FHZ_data/FHZ HG Palex 1.txt',
    'FHZ_data/FHZ HG Palex 2.txt',
    'FHZ_data/FHZ HG Palex 3.txt'
]

process_file(files[0])
process_file(files[1])
process_file(files[2])
process_file(files[3])


def masks (x, y) -> dict:
    return {
    'FHZ_data/FHZ Neon Palex.txt': (x < 8.8) & (x > 3),
    'FHZ_data/FHZ HG Palex 1.txt': (x < 5.5) & (x > 0.8),
    'FHZ_data/FHZ HG Palex 2.txt': (x < 8.2) & (x > 0.5),
    'FHZ_data/FHZ HG Palex 3.txt': (x < 5.2) & (x > 0.5) & (y < 8)
}


for file in files:
    x = data[file][1]
    y = data[file][2]

    mask = (y > 0)
    mask &= masks(x, y)[file]

    x = x[mask]
    y = y[mask]

    x_to_ys = defaultdict(list)
    x_to_weights = defaultdict(list)

    for xi, yi in zip(x, y):
        # Here, we treat each occurrence as weight 1. If you already have different weights, use them.
        x_to_ys[xi].append(yi)
        x_to_weights[xi].append(1)  # or another weight if available

    # Compute the weighted average for each unique x
    xdata_unique = []
    ydata_unique = []
    weights_unique = []

    for xi in sorted(x_to_ys.keys()):
        ys = np.array(x_to_ys[xi])
        ws = np.array(x_to_weights[xi])
        weighted_avg = np.average(ys, weights=ws)
        total_weight = np.sum(ws)
        xdata_unique.append(xi)
        ydata_unique.append(weighted_avg)
        weights_unique.append(total_weight)

    xdata_unique = np.array(xdata_unique)
    ydata_unique = np.array(ydata_unique)
    weights_unique = np.array(weights_unique)

    # Now, fit the spline using the aggregated data.
    spline = UnivariateSpline(xdata_unique, ydata_unique, k=3, s=200000, w=weights_unique)

    plot_data(
        datasets= [
            {
            'xdata': x,
            'ydata': y,
            'fit_xdata': np.linspace(min(x), max(x), 1000),
            'fit': spline(np.linspace(min(x), max(x), 1000)),
            'label': 'Data',
            'line': 'None',
            }
        ],
        filename=f'Plots/{file.split("/")[-1].split(".")[0].replace(' Palex', '')}.pdf',
        plot=False
    )

    spline_der = spline.derivative()
    spline_der2 = spline.derivative(n=2)

    roots = PPoly.from_spline(spline_der._eval_args).roots()
    roots = np.unique(roots) 

    maxima = [r for r in roots if spline_der2(r) < 0]

    print('\n')
    print('_'*50)
    print(f'File: {file}')
    print(f'Maxima: {maxima}')
    print(f'Maxima distances: {np.diff(maxima)}')
    print(f'Average distance: {np.mean(np.diff(maxima))}')

    plot_data(
        datasets= [
            {
            'xdata': np.linspace(0, len(maxima) - 1, len(maxima)),
            'ydata': maxima,
            'label': 'Maxima',
            'line': 'None',
            }
        ],
        filename=f'Plots/{file.split("/")[-1].split(".")[0].replace(' Palex', '')}_maxima.pdf',
        plot=False
    )



