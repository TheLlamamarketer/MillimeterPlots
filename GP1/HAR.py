import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.signal import find_peaks

# Load the data, skipping the first 100 rows
file_path = 'C:/Users/alexa/OneDrive/Documents/Uni/HAR_7.6.24/2.80-7.txt'
data = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=5)

data.columns = ['Time', 'Amplitude', 'Frequency', 'Space']
data = data.drop(columns=['Frequency', 'Space'])

# Further trim the data to remove initial rows with potential noise
trim_rows = int(0 * 100)
data = data.iloc[trim_rows:]

# Convert columns to float and remove rows with NaN values
time = data['Time'].astype(float)
amplitude = data['Amplitude'].astype(float)
time = time[amplitude.notna()]
amplitude = amplitude[amplitude.notna()]

# Interpolation function
interp_func = interp1d(time, amplitude, kind='linear')

# Define the fitting function: a combination of sine and linear functions
def func1(x, a, b):
    return np.abs(a) * np.sin(b * x) 

def func2(x, a, b, c, d, e):
    return np.abs(a) * np.sin(b * x + c) - d * x + e

def residuals1(params, x, y):
    return y - func1(x, *params)

def residuals2(params, x, y):
    return y - func2(x, *params)

p0 = [0.05, 1]

p1, _ = leastsq(residuals1, p0, args=(time, amplitude))
p2 = np.concatenate((p1, [-1, 0, 0]))

params, _ = leastsq(residuals2, p2, args=(time, amplitude))
res = residuals2(params, time, amplitude)
rms = np.sqrt(np.mean(res**2))

# Extract fitted parameters
a, b, c, d, e = params
print(f"{a:.6f}\t{b:.6f}")


# Generate new time values for interpolation matching original data points
time_new = np.linspace(time.min(), time.max(), num=len(time))
amplitude_new = interp_func(time_new)

# Plotting the fitted curve
plt.figure(figsize=(12, 8))
plt.plot(time_new, amplitude_new, '-', label='Interpolated Curve', color='red', linewidth=1)
plt.plot(time, amplitude, 'o', label='Original Data', markersize=3, color='blue')
plt.plot(time_new, func1(time_new, *p1), '-', label='Initial Guess Curve', color='orange', linewidth=2)
plt.plot(time_new, func2(time_new, *params), '-', label='Fitted Curve', color='green', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (m)', fontsize=14)
plt.title('Time vs. Amplitude', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


peaks, properties = find_peaks(amplitude, distance=15)
peaks_2, properties_2 = find_peaks(-amplitude, distance=15)
original_peaks = time.index[peaks]
original_peaks_2 = time.index[peaks_2]


def print_csv_format(time_data, amplitude_data):
    csv_data = "Time,Amplitude\n"
    for t, a in zip(time_data, amplitude_data):
        csv_data += f"{t},{a}\n"
    print(csv_data)

print_csv_format(time.loc[original_peaks].values, amplitude.iloc[peaks].values)
print_csv_format(time.loc[original_peaks_2].values, amplitude.iloc[peaks_2].values)

plt.figure(figsize=(12, 8))
plt.plot(time, amplitude, label='Amplitude')
plt.plot(time.loc[original_peaks], amplitude.iloc[peaks], "x", label='Peaks')
plt.plot(time.loc[original_peaks_2], amplitude.iloc[peaks_2], "x", label='Peaks')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('Amplitude with Peaks', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


# Plotting residuals
# plt.figure(figsize=(12, 8))
# plt.plot(time, res, '-', label='Residuals', color='red', linewidth=2)
# plt.xlabel('Time (s)', fontsize=14)
# plt.ylabel('Residual Amplitude (m)', fontsize=14)
# plt.title('Residuals vs. Time', fontsize=16)
# plt.legend()
# plt.grid(True)
# plt.show()