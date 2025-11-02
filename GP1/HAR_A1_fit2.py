import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Provided data
x_data = np.array([1.14, 2.85, 4.54, 6.24, 7.94, 9.64, 11.34, 13.03, 14.73, 16.41, 18.07, 19.673])
y_data = np.array([-2.46510402, -2.61729584, -2.81341072, -2.99573227, -3.19418321, -3.41124772, -3.64965874, -3.9633163, -4.34280592, -4.82831374, -5.52146092, -6.90775528])

# Define the logarithmic function with an offset and a linear term
def log_linear_offset_func(x, a, b, c, d):
    return a * np.log(b * x + d) + c

# Ensure positive argument for log by setting an initial guess for c that makes b*x + c > 0
initial_guess = [1, 10, 3, 10]

# Fit the data
params, _ = curve_fit(log_linear_offset_func, x_data, y_data, p0=initial_guess)

# Extract parameters
a, b, c, d = params

# Generate fitted y data
y_fit = log_linear_offset_func(x_data, a, b, c, d)

print(f"Fitted parameters:\na: {a}\nb: {b}\nc: {c} \nd: {d}")

# Plot the data and the fit
plt.figure(figsize=(12, 8))
plt.plot(x_data, y_data, 'o', label='Original Data', markersize=5, color='blue')
plt.plot(x_data, y_fit, '-', label='Fitted Curve', color='red')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.title('Logarithmic with Offset and Linear Term Fit', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()