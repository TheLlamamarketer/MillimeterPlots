import numpy as np
import matplotlib.pyplot as plt

# Data
amplitude = np.array([0.0125, 0.008, 0.015, 0.012, 0.015428571, 0.038, 0.029178571, 0.036, 0.037, 0.032, 0.036090909, 0.278479409, 0.276896116, 0.278732675, 0.276618843, 0.104304084, 0.103838517, 0.107024879, 0.103635271, 0.106234305, 0.033, 0.026, 0.013619048, 0.01325, 0.013, 0.013])
amplitude_error = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000235585, 0.000291277, 0.000230848, 0.000746703, 0.000820186, 0.000654195, 0.001302184, 0.000215795, 0.00034844, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
angular_frequency = np.array([0.972918933, 0.96848918, 1.380296872, 1.343637596, 2.152683138, 3.112689112, 3.133266289, 3.095165176, 3.106642921, 3.111187863, 3.108179722, 3.645307406, 3.647158258, 3.646530814, 3.645616494, 3.924487955, 3.926377321, 3.922904458, 3.9266679, 3.92234149, 4.422831716, 4.633678504, 5.469584598, 5.479941282, 5.494881197, 5.500843168])
angular_frequency_error = np.array([0.000410021, 0.000406296, 0.001072063, 0.001284989, 0.002007305, 0.004049558, 0.002979534, 0.002856047, 0.004033841, 0.004542801, 0.004745017, 0.002707843, 0.005466177, 0.004232626, 0.00456187, 0.003466579, 0.005419101, 0.005546491, 0.003419518, 0.004508166, 0.010678531, 0.012081669, 0.010516032, 0.010823174, 0.009334989, 0.009183586])

# Theoretical resonance curve
def resonance_curve(omega, omega_0, A_max, gamma):
    return A_max / np.sqrt((omega_0**2 - omega**2)**2 + (gamma * omega)**2)

# Parameters for the theoretical curve (can be adjusted)
omega_0 = 3.7  # Resonance angular frequency
A_max = 0.15  # Maximum amplitude
gamma = 0.1  # Damping coefficient

# Generate theoretical curve
omega_theory = np.linspace(min(angular_frequency), max(angular_frequency), 5000)
amplitude_theory = resonance_curve(omega_theory, omega_0, A_max, gamma)

# Plot data with error bars
plt.errorbar(angular_frequency, amplitude/2, xerr=angular_frequency_error, yerr=amplitude_error, fmt='o', label='Data', ecolor='red', capsize=2, markersize=3, color='blue')

# Plot theoretical curve
plt.plot(omega_theory, amplitude_theory, label='Theoretical Resonance Curve')

# Plot settings
plt.xlabel('Angular Frequency (rad/s)')
plt.ylabel('Amplitude')
plt.title('Resonance Curve of a Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()