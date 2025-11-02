import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Data with actual distances to point P
distances_to_P = np.array([0.41, 0.47, 0.55, 0.63, 0.71, 0.718, 0.73, 0.75, 0.77, 0.79, 0.87])
periods_P = np.array([1.885, 1.893, 1.916, 1.948, 1.987, 1.991, 1.997, 2.008, 2.02, 2.031, 2.078])
periods_Q = np.array([1.956, 1.954, 1.957, 1.97, 1.992, 1.995, 1.999, 2.007, 2.015, 2.025, 2.07])

# Fit a 4th degree polynomial to each set of data
poly_P = Polynomial.fit(distances_to_P, periods_P, 4)
poly_Q = Polynomial.fit(distances_to_P, periods_Q, 4)

# Find the intersection by subtracting the polynomials and finding the roots
intersection_poly = poly_P - poly_Q
roots = intersection_poly.roots()

# Filter roots to find the intersection in the range of distances
real_roots = roots[np.isreal(roots)].real
intersection_points = [(root, poly_P(root)) for root in real_roots if min(distances_to_P) <= root <= max(distances_to_P)]

# Plotting
plt.figure(figsize=(10, 6))
x_vals = np.linspace(min(distances_to_P), max(distances_to_P), 400)
plt.plot(x_vals, poly_P(x_vals), label='Period at P')
plt.plot(x_vals, poly_Q(x_vals), label='Period at Q')
plt.scatter(distances_to_P, periods_P, color='blue', label='Data Points P')
plt.scatter(distances_to_P, periods_Q, color='red', label='Data Points Q')
for pt in intersection_points:
    plt.plot(pt[0], pt[1], 'go', label=f'Intersection Point ({pt[0]:.2f}, {pt[1]:.4f})')  # mark the intersection points
plt.xlabel('Distance to P (m)')
plt.ylabel('Period (s)')
plt.legend()
plt.title('Period at P and Q with Intersection Points')
plt.grid(True)
plt.show()

intersection_points