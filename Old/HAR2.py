import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

amplitude = np.array([0.036090909,  0.278732675,  0.104304084, 0.033])
angular_frequency = np.array([ 3.108179722, 3.646530814, 3.924487955,  4.422831716])

# Define the driven harmonic oscillator function
def driven_oscillator(omega, F0, m, d, w0):
    return F0/m * 1 / (2*d*w0 * np.sqrt(1 + (abs( w0 - omega)/d)**2))

initial_guess = [0.8, 6, 0.01, 3.6]
bounds = (0, [np.inf, np.inf, np.inf, np.inf])

# Fit the model to the data
params, params_covariance = curve_fit(driven_oscillator, angular_frequency, amplitude, p0=initial_guess, bounds=bounds)

# Extract fitted parameters
F0, m, d, w0 = params

# Print fitted parameters
print(f"Fitted parameters:\nF0: {F0}\nm: {m}\nd: {d}\nw0: {w0}")

# Generate fitted data
omega_fit = np.linspace(angular_frequency.min(), angular_frequency.max(), 1000)
amplitude_fit = driven_oscillator(omega_fit, *params)

# Plot the data and the fit
plt.figure(figsize=(12, 8))
plt.scatter(angular_frequency, amplitude, label='Original Data', color='blue')
plt.plot(omega_fit, amplitude_fit, label='Fitted Curve', color='red')
plt.xlabel('2π/T (angular frequency)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('Driven Harmonic Oscillator Fit', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()