import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.optimize import leastsq, curve_fit

# Load the data, skipping the first 100 rows
file_path = 'C:/Users/alexa/OneDrive/Documents/Uni/HAR_7.6.24/3.7nR.txt'
data = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=5)

data.columns = ['Time', 'Amplitude', 'Frequency', 'Space']
data = data.drop(columns=['Frequency', 'Space'])

trim_rows = int(0* 100)
data = data.iloc[trim_rows:]
cut_rows = int(0.1* 100)
data = data.iloc[:-cut_rows]


time = data['Time'].astype(float)
amplitude = data['Amplitude'].astype(float)
time = time[amplitude.notna()]
amplitude = amplitude[amplitude.notna()]

time = time - time.min()

segment_indices = [0, int(97* 100), int(103.2* 100), len(time)]

colors = ['royalblue', 'mediumseagreen', 'lightcoral']
labels = ['Rebalanzierungsphase', 'Stabilisierte Phase', 'Exponentielle Abfallphase']

plt.figure(figsize=(12, 8))

# Plot each segment with a different color
for i in range(len(segment_indices) - 1):
    start_idx = segment_indices[i]
    end_idx = segment_indices[i + 1]
    plt.plot(time[start_idx:end_idx], amplitude[start_idx:end_idx], '-', label=labels[i], color=colors[i])

plt.xlabel('Zeit (s)', fontsize=14)
plt.ylabel('Amplitude (m)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()