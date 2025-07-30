import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
import cv2
from plotting import plot_data
from help import *


# Load the image in grayscale
files = ['0', '1', '3', '5', '7', '9', '11', '100000']
files2 = ['G1', 'G2', 'G3', 'G4', 'G5_2']
files3 = ['a3_G1', 'a3_G2', 'a3_G3', 'a3_G4', 'a3_G5']

files0 = ['1']

for file in files:
    img = cv2.imread(f'FOU_data/{file}.png', cv2.IMREAD_UNCHANGED)
    if file == '100000':
        file_name = '\\infty'
    elif '_2' in file:
        file_name = file[:-2]
    elif 'a3' in file:
        file_name = file[3:]
    else:
        file_name = file

    #print(img.shape, img.dtype, img.min(), img.max())

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

    def autocorrelation(x):
        n = len(x)
        x = x - np.mean(x)
        #result = np.correlate(x, x, mode='full')
        #return result[result.size // 2:] / (np.var(x) * np.arange(n, 0, -1))
        result = np.array([np.sum(x[:n-k] * x[k:]) for k in range(n)]) / np.sqrt(np.array([np.sum(x[:n-k] ** 2) * np.sum(x[k:] ** 2) for k in range(n)]))
        return result

    def find_extrema(x):
        maxima = [0]
        minima = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                maxima.append(i)
            if x[i] < x[i - 1] and x[i] < x[i + 1]:
                minima.append(i)
        return maxima, minima

    #autocor_res = autocorrelation(average_slice)
    #maxima, minima = find_extrema(autocor_res)

    #print(f"Distances between maxima: {np.diff(maxima)}")
    #print(f"Distances between minima: {np.diff(minima)}")

    plot_data(
        datasets=[
            {
                'xdata': t_data,
                'ydata': average_slice,
                'label': f'max Beugungsordnung ${file_name}$',
                'line': '-',
                'marker': None,
                'confidence': [(average_slice - std_slice , average_slice + std_slice), (average_slice - 2*std_slice, average_slice + 2*std_slice)]
            },
        ],
        x_label='Pixel Spalte',
        y_label='Summe der Intensitäten',
        title=f'Intensitätsprofil mit Gitter 1 und max. Beugungsordnung ${file_name}$',
        filename=f'Plots/FOU_{file}.pdf',
        width=25,
        height=9,
        plot=False,
    )

    #plot_data(
    #    datasets=[
    #        {
    #            'xdata': t_data,
    #            'ydata': autocor_res,
    #            'label': 'Autokorrelation',
    #            'line': '-',
    #            'marker': None,
    #        },
    #    ],
    #    x_label='Verzögerung',
    #    y_label='Autokorrelation',
    #    title='Autokorrelation des Intensitätsprofils',
    #    filename=f'Plots/FOU_{file}_autocorrelation.pdf',
    #    width=25,
    #    height=10,
    #    plot=False,
    #)



    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=60)  # Decreased dpi for lower quality
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_green', ['black', 'green', 'lime'], N=256)
    plt.imshow(red_chan, cmap=cmap_yell)
    plt.imshow(green_working, cmap=cmap)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])  
    plt.savefig(f'FOU_data2/{file}.png', bbox_inches='tight', pad_inches=0, format='png', dpi=60)  # Decreased dpi for lower quality
    #plt.show()
    plt.close()

# Fourier series approximation of a square wave
import numpy as np
import matplotlib.pyplot as plt

# Time domain
t = np.linspace(0, 8*np.pi, 1000)
orders = [0, 1, 3, 5, 7, 11, 99] 


# Create subplots with adjusted figure size
fig, axes = plt.subplots(len(orders), 1, figsize=(8, 12), constrained_layout=True)

for idx, N in enumerate(orders):
    # Initialize the square wave approximation
    square_wave = np.zeros_like(t) 
    # Sum over the odd harmonics up to order N
    for n in range(1, N + 1, 2):
        square_wave += (0.5/(np.pi * n)) * np.sin(n * t) 
    # Plot the approximation
    axes[idx].plot(t, square_wave + 0.125, label=f'Ordnung {N}')
    axes[idx].set_ylim(-0.1, 0.4)
    axes[idx].grid(True)
    axes[idx].legend(loc='upper right')

axes[-1].set_xlabel('x')  # Add x-axis label to the last subplot

# Remove plt.tight_layout() if using constrained_layout
# plt.tight_layout()
plt.savefig('Plots/Square.pdf', format='pdf', bbox_inches='tight')
plt.close()

