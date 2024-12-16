import numpy as np
import matplotlib.pyplot as plt
import cv2
from plotting_minus import plot_data
from lmfit import Model
from scipy.signal import square
from help import *


# Load the image in grayscale
img = cv2.imread('FOU_data/G5.png', cv2.IMREAD_UNCHANGED)
print(img.shape, img.dtype, img.min(), img.max())

red_chan = img[:,:,2]
cmap_yell = plt.cm.colors.LinearSegmentedColormap.from_list('custom_yellow', ['black', 'yellow'], N=256)


green_chan = img[:,:,1]
arrow_mask = (green_chan > 50) & (red_chan > 50)

green_working = green_chan.astype(float)
green_working[arrow_mask] = np.nan

slices = []
range_slices = (0, 1280)

slices = []
for i in range(range_slices[0], range_slices[1]):
    vertical_slice = green_working[:, i]
    normalized_slice = vertical_slice / 255.0 
    slices.append(normalized_slice)

average_slice = np.nanmean(slices, axis=0)
std_slice = np.nanstd(slices, axis=0)

t_data = np.arange(len(average_slice))

data = {
    'G5':[6.5, 22.5, 37, 50.5, 64],
    'dG5':[1, 2, 2, 3, 3],
    'lambda5': 63.33,
    'G4':[5.5, 15.5, 26.5, 36, 47],
    'dG4':[1, 2, 1.5, 2.5, 2.5],
    'lambda4': 92.7,
    'G3':[4, 11.5, 19, 26, 33.5],
    'dG3':[0.75, 1, 1.5, 2, 2.5],
    'lambda3': 130.57,
    'G2':[3, 8, 14, 18.5, 24, 29, 34],
    'dG2':[0.5, 2, 1, 2, 1.5, 2, 2],
    'lambda2': 185,
    'G1':[2, 6, 9, 13, 16.5, 20.5, 24.5, 28],
    'dG1':[0.5, 0.5, 1.5, 1.5, 2, 1.5, 2, 2],
    'lambda1': 260.67,
}

for key in data:
    if isinstance(data[key], list):
        data[key] = np.array(data[key])
        data[key] = data[key]/2


plot_data(
    datasets=[
        {
            'xdata': t_data,
            'ydata': average_slice,
            'label': 'Intensity Profile',
            'line': '-',
            'marker': None,
            'confidence': [(average_slice - std_slice , average_slice + std_slice), (average_slice - 2*std_slice, average_slice + 2*std_slice)]
        },
        #{
        #    'xdata': t_data,
        #    'ydata': fitted_wave,
        #    'label': 'Fitted Square Wave',
        #    'line': '--',
        #    'marker': None,
        #    'confidence': None,
        #},
    ],
    x_label='Pixel Row',
    y_label='Intensity',
    title='Intensity Profile with Error Bounds',
    filename='Plots/FOU_G2.pdf',
    width=25,
    height=10,
    plot=False,
)


lambdas = [data['lambda1'], data['lambda2'], data['lambda3'], data['lambda4'], data['lambda5']]

datasets = []
max_length = max(len(data['G1']), len(data['G2']), len(data['G3']), len(data['G4']), len(data['G5']))

for i in range(max_length):
    xdata = []
    ydata = []
    yerr = []
    for G_key, lambda_value in zip(['G1', 'G2', 'G3', 'G4', 'G5'], lambdas):
        G_values = data[G_key]
        if len(G_values) > i:
            ydata.append(G_values[i])
            xdata.append(lambda_value)
            yerr.append(data[f'd{G_key}'][i])

    datasets.append({
        'xdata': xdata,
        'ydata': ydata,
        'y_error': yerr,
        'line': 'None',
        'marker': '.'
    })

plot_data(
    datasets=datasets,
    x_label='grating spacing',
    y_label='Grating frequency',
    title='Grating frequency vs grating spacing',
    filename='Plots/FOU_data1.pdf',
    width=25,
    height=10,
    plot=False,
)

# add the log of xdata and ydata to replace the datasets

datasets = []
params = []
max_length = max(len(data['G1']), len(data['G2']), len(data['G3']), len(data['G4']), len(data['G5']))

for i in range(max_length):
    xdata = []
    ydata = []
    yerr = []
    for G_key, lambda_value in zip(['G1', 'G2', 'G3', 'G4', 'G5'], lambdas):
        G_values = data[G_key]
        if len(G_values) > i:
            ydata.append(G_values[i])
            xdata.append(lambda_value)
            yerr.append(data[f'd{G_key}'][i])
    
    if len(xdata) > 1 and len(ydata) > 1:
        result = linear_fit(np.log(xdata), np.log(ydata), np.abs(np.array(yerr) / np.array(ydata)), model="linear")
        fit = result.eval(x=np.log(xdata))
        confidence = calc_CI(result, np.log(xdata))
        param = extract_params(result)
        params.append(param)
    else:
        fit = None
        confidence = None

    log_yerr_lower = np.log(ydata) - np.log(np.array(ydata) - np.array(yerr))
    log_yerr_upper = np.log(np.array(ydata) + np.array(yerr)) - np.log(ydata)

    datasets.append({
        'xdata': np.log(xdata),
        'ydata': np.log(ydata),
        'y_error': (log_yerr_lower, log_yerr_upper),
        'fit': fit,
        'line': 'None',
        'marker': '.',
        'confidence': confidence
    })

for param in params:
    b, db, _ = round_val(param['b'][0], param['b'][1])
    if db != 0:
        print(f"Steigung b = {param['b'][0]}")

plot_data(
    datasets=datasets,
    x_label='Log of grating spacing',
    y_label='Log of Grating frequency',
    title='Log-Log Plot of Grating frequency vs grating spacing',
    filename='Plots/FOU_data2.pdf',
    width=25,
    height=10,
    plot=False,
)


# Plot the image with the slice being shown as a line
plt.figure(figsize=(10, 10))
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_green', ['black', 'green', 'lime'], N=256)
plt.imshow(red_chan, cmap=cmap_yell)
plt.imshow(green_working, cmap=cmap)
plt.axis('off')
plt.show()
