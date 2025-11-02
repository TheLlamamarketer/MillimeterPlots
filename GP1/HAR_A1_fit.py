import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.signal import find_peaks

# Load the data, skipping the first 100 rows
file_path = 'C:/Users/alexa/OneDrive/Documents/Uni/HAR_7.6.24/1.3.2.txt'
data = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=5)

data.columns = ['Time', 'Amplitude', 'Frequency', 'Space']
data = data.drop(columns=['Frequency', 'Space'])

trim_rows = int(0.5 * 100)
cut_rows = int(1* 100)

data = data.iloc[trim_rows:]
data = data.iloc[:-cut_rows]

time = data['Time'].astype(float)
amplitude = data['Amplitude'].astype(float)
time = time[amplitude.notna()]
amplitude = amplitude[amplitude.notna()]

time = time - time.min()

def func1(x, a, b):
    return a * np.cos(b * x)

def func2(x, a, b, c, d, e):
    return a* np.exp(-d * x) * np.cos(b * x + e) + c

def func3(x, a, b, c, d, e, f):
    return (-a*x**2+c)*np.exp(-d * x) * np.cos(b * x + e) + f

def residuals1(params, x, y):
    return y - func1(x, *params)

def residuals2(params, x, y):
    return y - func2(x, *params)

def residuals3(params, x, y):
    return y - func3(x, *params)

p0 = [amplitude.max(), 3.8]
p1, _ = leastsq(residuals1, p0, args=(time, amplitude))
p2 = np.concatenate((p1, [0.04, 0.09, 0]))
p3, _ = leastsq(residuals2, p2, args=(time, amplitude))
p4 = np.concatenate((p1, [1, 0.06, 0, -0.002]))
params, _ = leastsq(residuals3, p4, args=(time, amplitude))

res = residuals3(params, time, amplitude)
rms = np.sqrt(np.mean(res**2))

# Extract fitted parameters
a2, b2, c2, d2, e2 = p3
a3, b3, c3, d3, e3, f3 = params

print(f"\nFit 2:\t{b2:.6f}\t{d2:.6f} \t\t{c2:.6f} \t{a2:.6f} \t{e2:.6f}")
print(f"Fit 3:\t{b3:.6f}\t{d3:.6f} \t\t{c3:.6f} \t{a3:.6f} \t{e3:.6f} \t{f3:.6f}")


time_new = np.linspace(time.min(), time.max(), num=len(time))

# Plotting the fitted curves
plt.figure(figsize=(12, 8))
plt.plot(time, amplitude, 'o', label='Original Data', markersize=3, color='blue')
plt.plot(time_new, func2(time_new, *p3), '-', label='Fitted Curve 2', color='orange', linewidth=2)
plt.plot(time_new, func3(time_new, *params), '-', label='Fitted Curve 3', color='green', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (m)', fontsize=14)
plt.title('Time vs. Amplitude', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time, np.log(amplitude), 'o', label='Original Data', markersize=3, color='blue')
plt.plot(time_new, np.log(func2(time_new, *p3)), '-', label='Fitted Curve 2', color='orange', linewidth=2)
plt.plot(time_new, np.log(func3(time_new, *params)), '-', label='Fitted Curve', color='green', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Log(Amplitude)', fontsize=14)
plt.title('Log(Amplitude) vs. Time', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


log_amplitude = np.log(amplitude)
log_amplitude_2 = np.log(-amplitude)
peaks, properties = find_peaks(log_amplitude, distance=25, prominence=0.1)
peaks_2, properties_2 = find_peaks(log_amplitude_2, distance=25, prominence=0.1)
original_peaks = time.index[peaks]
original_peaks_2 = time.index[peaks_2]


print(time.loc[original_peaks].values)
print(log_amplitude.iloc[peaks].values)

print(time.loc[original_peaks_2].values)
print(log_amplitude_2.iloc[peaks_2].values)

plt.figure(figsize=(12, 8))
plt.plot(time, log_amplitude, label='Log(Amplitude)')
plt.plot(time, log_amplitude_2, label='Log(Amplitude)')
plt.plot(time.loc[original_peaks], log_amplitude.iloc[peaks], "x", label='Peaks')
plt.plot(time.loc[original_peaks_2], log_amplitude_2.iloc[peaks_2], "x", label='Peaks')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Log(Amplitude)', fontsize=14)
plt.title('Log(Amplitude) with Peaks', fontsize=16)
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