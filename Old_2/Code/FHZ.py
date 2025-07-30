import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from collections import defaultdict
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *
from scipy.interpolate import UnivariateSpline, PPoly

print('\n')

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

def intervals (x) -> dict:
    return {
    'FHZ_data/FHZ Neon Palex.txt': (),
    'FHZ_data/FHZ HG Palex 1.txt': (x < 10) & (x > 2),
    'FHZ_data/FHZ HG Palex 2.txt': (x < 14) & (x > 3),
    'FHZ_data/FHZ HG Palex 3.txt': (x < 8) & (x > 1)
}

names = [
    'Neon',
    'Hg 190°C',
    'Hg 210°C',
    'Hg 150°C'
]

datasets_together = []
datasets_max_together = []



for j, file in enumerate(files):
    x = data[file][1]
    y = data[file][2]
    sy = 0.07874

    mask = (y > 0)
    mask &= masks(x, y)[file]

    x = x[mask]*10.0
    y = y[mask]

    x_to_ys = defaultdict(list)

    for xi, yi in zip(x, y):
        x_to_ys[xi].append(yi)

    xdata_unique = []
    ydata_unique = []
    n = []
    yerr_unique = []

    for xi in sorted(x_to_ys.keys()):
        ys = np.array(x_to_ys[xi])
        avg = np.mean(ys)
        ni = np.sum(len(ys))

        xdata_unique.append(xi)
        ydata_unique.append(avg)
        n.append(ni)

    xdata_unique = np.array(xdata_unique)
    ydata_unique = np.array(ydata_unique)
    n = np.array(n)

    # Now, fit the spline using the aggregated data.
    spline = UnivariateSpline(xdata_unique, ydata_unique, k=3, s=200000, w=n)


    max_positions = [] 

    x_fit = np.linspace(xdata_unique.min(), xdata_unique.max(), 1000)
    splines = []

    for i in range(1000):
        y_sim = np.random.normal(ydata_unique, sy / np.sqrt(n), size=ydata_unique.shape)
        
        spline_sim = UnivariateSpline(xdata_unique, y_sim, k=3, s=200000, w=n)
        splines.append(spline_sim(x_fit))
        
        try:
            roots = PPoly.from_spline(spline_sim.derivative()._eval_args).roots()
            roots = np.unique(roots)
            maxima = np.array([r for r in roots if np.all(np.array(spline_sim.derivative(n=2)(r)) < 0)])
        except Exception as e:
            maxima = np.array([np.nan])
        
        max_positions.append(maxima)

    splines = np.array(splines)

    mean_spline = np.mean(splines, axis=0)
    #lower_bound = np.percentile(splines, 1, axis=0)
    #upper_bound = np.percentile(splines, 99, axis=0)

    # Pad the max_positions lists with NaN values to ensure they are of the same length
    max_length = max(len(maxima) for maxima in max_positions)
    max_positions = np.array([np.pad(maxima, (0, max_length - len(maxima)), constant_values=np.nan) for maxima in max_positions])

    mean_max = np.nanmean(max_positions, axis=0)
    std_max = np.nanstd(max_positions, axis=0)

    numbers = np.linspace(1, len(mean_max), len(mean_max))

    mask = intervals(numbers)[file]
    new_maxima = mean_max[mask]
    new_numbers = numbers[mask]

    result = lmfit(new_numbers, new_maxima)

    h = 6.62607015e-34
    c = 299792458
    e = 1.602176634e-19

    print(f'File: {file}')
    print(f"E = $({print_round_val(result.params['b'].value, result.params['b'].stderr)})eV$")

    wavelength = h*c /result.params['b'].value /e
    print(f"\\lambda_{{{names[j].replace('Hg ', '').replace('on', '')}}} = ({print_round_val(wavelength*1e9, result.params['b'].stderr * h*c /result.params['b'].value**2 /e*1e9)})\\, \\mathrm{{nm}}")
    print(f"f_{{{names[j].replace('Hg ', '').replace('on', '')}}} = ({print_round_val(result.params['b'].value / h * e * 1e-15, result.params['b'].stderr / h * e * 1e-15)})\\, 10^{{15}} \\mathrm{{Hz}}")
    print('\n')

    datasets= [{
        'xdata': x,
        'ydata': y,
        'fit_xdata': x_fit,
        'fit': mean_spline,
        #'confidence': [(lower_bound, upper_bound)],
        'label': f'{names[j]}',
        'line': 'None',
        'color_group': j,
    }]

    datasets_together.extend(datasets)

    datasets_max = [{
        'xdata': numbers,
        'ydata': mean_max,
        'yerr': std_max,
        'label': f'Maxima {names[j]}',
        'line': 'None',
        'fit': result.eval(x=numbers),
        'confidence': calc_CI(result, numbers, sigmas=[2]),
    }]
    datasets_max_together.extend(datasets_max)

plot_data(
    datasets = datasets_max_together[1:],
    ylabel= r'$E_{kin} / eV$',
    xlabel= r'Peaknummer \#',
    filename=f'Plots/FHZ All max.pdf',
    plot=False
)

plot_data(
    datasets= datasets_max_together[0],
    ylabel= r'$E_{kin} / eV$',
    xlabel= r'Peaknummer \#',
    filename=f'Plots/{files[0].split("/")[-1].split(".")[0].replace(" Palex", "")} max.pdf',
    plot=False
)


plot_data(
    datasets = datasets_together[1:],
    ylabel= r'$U_A / V$',
    xlabel= r'$U_B / V$',
    filename=f'Plots/FHZ All.pdf',
    plot=False
)

for file, datasets in zip(files, datasets_together):
    plot_data(
        datasets = datasets,
        ylabel= r'$U_A / V$',
        xlabel= r'$U_B / V$',
        filename=f'Plots/{file.split("/")[-1].split(".")[0].replace(" Palex", "")}.pdf',
        plot=False
    )
