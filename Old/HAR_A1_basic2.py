import numpy as np
import matplotlib.pyplot as plt

# Define the data sets
data_sets = {
    'T_3_2': [0, 1.705, 3.405, 5.105, 6.805],
    'A_3_2': [0.076, 0.052, 0.033, 0.021, 0.011],
    'T_3_1': [0, 1.705, 3.405, 5.105, 6.805],
    'A_3_1': [0.075, 0.051, 0.033, 0.02, 0.01],
    'T_2_2': [0, 1.7, 3.4, 5.1, 6.795, 8.495, 10.195, 11.9],
    'A_2_2': [0.085, 0.073, 0.061, 0.051, 0.042, 0.034, 0.026, 0.02],
    'T_2_1': [0, 1.71, 3.405, 5.105, 6.805, 8.505, 10.195, 11.895],
    'A_2_1': [0.085, 0.073, 0.06, 0.05, 0.041, 0.033, 0.026, 0.019],
    'T_1_2': [0, 1.695, 3.39, 5.1, 6.79, 8.48, 10.19, 11.885],
    'A_1_2': [0.085, 0.075, 0.064, 0.055, 0.045, 0.037, 0.03, 0.024],
    'T_1_1': [0, 1.7, 3.4, 5.105, 6.8, 8.5, 10.195, 11.89],
    'A_1_1': [0.084, 0.073, 0.064, 0.055, 0.047, 0.04, 0.033, 0.026],
    'T_0': [0, 1.7, 3.4, 5.1, 6.8, 8.5, 10.2, 11.9],
    'A_0': [0.095, 0.089, 0.084, 0.078, 0.072, 0.067, 0.06, 0.055]
}

def func(T, A, c=0):
    A_offset = A + c
    A_offset = np.log(A_offset)
    m = np.sum((A_offset - np.mean(A_offset)) * (T - np.mean(T))) / np.sum((T - np.mean(T))**2)
    n = np.mean(A_offset) - m * np.mean(T)
    return m, n

def plot_data(T_key, A_key, c=0):
    T = np.array(data_sets[T_key])
    A = np.array(data_sets[A_key])
    
    # Compute the slope and intercept
    m, n = func(T, A, c)
    
    # Generate the linear fit line
    T_fit = np.linspace(min(T), max(T), 100)
    A_fit = m * T_fit + n

    RSS = np.sum((np.log(A+c) - (m * T + n))**2)

    dm= np.sqrt(RSS / (len(T) - 2) / np.sum((T - np.mean(T))**2))

    print(f"m = {-m:.5f} \pm {dm:.5f}")
    
    # Plot the actual data
    plt.scatter(T, np.log( A + c) , color='red', label='Actual Data')
    
    # Plot the linear fit
    plt.plot(T_fit, A_fit, label='Linear Fit')
    
    plt.xlabel('Period (T)')
    plt.ylabel('Amplitude (A)')
    plt.title('Linear Fit of Amplitude vs Period')
    plt.legend()
    plt.grid(True)
    plt.show()

    return RSS, m, dm

def plot_rss_vs_c(T_key, A_key, c_values):
    T = np.array(data_sets[T_key])
    A = np.array(data_sets[A_key])
    
    rss_values = []
    
    for c in c_values:
        m, n = func(T, A, c)
        RSS = np.sum((np.log(A+c)  - (m * T + n))**2)
        rss_values.append(RSS)
    
    # Plot RSS vs c
    plt.plot(c_values, rss_values, marker='o')
    plt.xlabel('Offset (c)')
    plt.ylabel('RSS')
    plt.title('RSS vs Offset (c)')
    plt.grid(True)
    plt.show()

# Example usage
plot_data('T_0', 'A_0', 0)


# Define a range of c values to test
c_values = np.linspace(0.01, 0.03, 1000)
plot_rss_vs_c('T_3_2', 'A_3_2', c_values)