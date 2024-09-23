import numpy as np
import math



def largest_power_of_10(n):
    if n != 0:
        exponent = math.floor(math.log10(abs(n)))
        return 10**exponent
    else:
        return 0

def find_clean_intervals(lower_bound, upper_bound, total_cm, min_increment=0.0001, max_increment=None):

    # normalize the input values to around 1, even if they are very large or very small
    largest_power = largest_power_of_10(max(abs(lower_bound), abs(upper_bound)))
    lower_bound /= largest_power
    lower_bound = round(lower_bound, 10)
    upper_bound /= largest_power
    upper_bound = round(upper_bound, 10)

    initial_interval = upper_bound - lower_bound
    current_interval = round(initial_interval, 4)
    clean_intervals = []

    # If max_increment is not provided, set it to 1/10 of the upper bound
    if max_increment is None: max_increment = upper_bound/10
    
    # Loop to find clean intervals
    while current_interval <= initial_interval + max_increment/largest_power:
        factor = 100 * (current_interval /total_cm )
        if round(factor, 8).is_integer():
            clean_intervals.append((current_interval, factor/100))
        current_interval += min_increment
        current_interval = round(current_interval, 10)
    
    # Display results if any clean intervals were found
    if clean_intervals:
        interval = upper_bound - lower_bound
        print(f"Initial interval: {round(interval*largest_power, 10)}")
        for result in clean_intervals:
            print(f"Interval: {round(result[0]*largest_power, 10)}, Units/cm: {round(result[1]*largest_power, 10)}")
    else:
        print("No clean intervals found.")


def find_interval(xdata, ydata, name=' ', xerror = None, yerror = None, width=20, height=28, max_increment_height=1, max_increment_width=1):
    print()
    print('------------------------------------------------------------')
    print(f'Starting to calculate intervals for width and height for {name}' )

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    xerror = np.array(xerror) if xerror is not None else np.zeros_like(xdata)
    yerror = np.array(yerror) if yerror is not None else np.zeros_like(ydata)

    width_interval = ((xdata - xerror).min(), (xdata + xerror).max())
    height_interval = ((ydata - yerror).min(), (ydata + yerror).max())

    print("Calculating intervals for Width:")
    find_clean_intervals(width_interval[0], width_interval[1], width, min_increment=0.0001, max_increment=max_increment_width)
    print('__________________________________________')
    print("\nCalculating intervals for Height:")
    find_clean_intervals(height_interval[0], height_interval[1], height, min_increment=0.0001, max_increment=max_increment_height)

    print(f'Finished calculating intervals for width and height for {name}')
    print('------------------------------------------------------------')
