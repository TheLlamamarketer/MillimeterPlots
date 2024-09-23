import numpy as np

# Given data
L_eff = 0.9941  # Fixed effective length in meters
distances_to_P = np.array([0.41, 0.47, 0.55, 0.63, 0.71, 0.718, 0.73, 0.75, 0.77, 0.79, 0.87])
periods_P = np.array([1.885, 1.893, 1.916, 1.948, 1.987, 1.991, 1.997, 2.008, 2.02, 2.031, 2.078])
periods_Q = np.array([1.956, 1.954, 1.957, 1.97, 1.992, 1.995, 1.999, 2.007, 2.015, 2.025, 2.07])

# Constants
d_fixed = 0.153  # fixed distance in meters
m_fixed = 1.0  # fixed mass in kg
m_moving = 1.4  # moving mass in kg

# Calculate center of mass
def calculate_center_of_mass(d_p):
    return (m_fixed * d_fixed + m_moving * d_p) / (m_fixed + m_moving)

# Calculate moment of inertia
def calculate_moment_of_inertia(d_p):
    I_fixed = m_fixed * d_fixed**2
    I_moving = m_moving * d_p**2
    return I_fixed + I_moving

# Calculate effective length
def calculate_effective_length(d_p):
    d_cm = calculate_center_of_mass(d_p)
    I = calculate_moment_of_inertia(d_p)
    return I / ((m_fixed + m_moving) * d_cm)

# Calculate periods using the given formula
def calculate_period(L_eff, g):
    return 2 * np.pi * np.sqrt(L_eff / g)

# Solve for g using the known L_eff and periods
def solve_for_g(T, L_eff):
    return 4 * np.pi**2 * L_eff / T**2

# Calculate periods for each configuration
def find_equal_period(distances, periods_P, periods_Q, L_eff):
    for i in range(len(distances)):
        for j in range(len(distances)):
            g_P = solve_for_g(periods_P[i], L_eff)
            g_Q = solve_for_g(periods_Q[j], L_eff)
            if np.isclose(g_P, g_Q, atol=1e-5):
                T_equal = (periods_P[i] + periods_Q[j]) / 2
                return T_equal, distances[i], distances[j]

# Find the equal period
T_equal, d_p_equal, d_q_equal = find_equal_period(distances_to_P, periods_P, periods_Q, L_eff)
print(f"The period where T_P = T_Q is {T_equal} s at distances d_P = {d_p_equal} m and d_Q = {d_q_equal} m")