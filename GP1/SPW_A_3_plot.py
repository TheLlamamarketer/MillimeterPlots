import numpy as np
import matplotlib.pyplot as plt

data = {
    't':  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180], #, 190, 200, 210], """[0,"""
    'T':  [24.1, 21.5, 19.9, 19.2, 18.9, 18.8, 18.9, 18.9, 18.9, 18.9, 18.9, 18.9, 18.9, 19, 19, 19, 19, 19] #, 19, 19, 19] """[31,"""
}

error = [i * 0.001 + 0.2 for i in data['T']]

def plot_data(data, error, ymin, ymax, xmax, xstep, xoffset, ystep, yoffset, grey_lower, grey_upper, filename):
    num_main_lines_x = 18
    num_main_lines_y = 28

    fig, ax = plt.subplots(figsize=(num_main_lines_x / 2.54, num_main_lines_y / 2.54))

    xmin = 0
    main_lines_x = np.arange(xmin, xmax, (xmax - xmin) / num_main_lines_x)
    main_lines_y = np.arange(ymin, ymax, (ymax - ymin) / num_main_lines_y)

    secondary_lines_x = np.arange(xmin, xmax, (xmax - xmin) / (num_main_lines_x * 2))
    secondary_lines_y = np.arange(ymin, ymax, (ymax - ymin) / (num_main_lines_y * 2))

    tertiary_lines_x = np.arange(xmin, xmax, (xmax - xmin) / (num_main_lines_x * 10))
    tertiary_lines_y = np.arange(ymin, ymax, (ymax - ymin) / (num_main_lines_y * 10))

    ax.set_xticks(np.arange(xmin + xoffset, xmax + 1, xstep))
    ax.set_yticks(np.arange(ymin + yoffset, ymax + 0.1, ystep))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_zorder(1)

    # tertiary grid lines
    for tick in tertiary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.1, zorder=0)
    for tick in tertiary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.1, zorder=0)

    # secondary grid lines
    for tick in secondary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.45, zorder=0)
    for tick in secondary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.45, zorder=0)

    # main grid lines
    for tick in main_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.7, zorder=0)
    for tick in main_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('#eb3107')
        spine.set_zorder(2)

    # Plot the black points first
    plt.errorbar(data['t'], data['T'], color='black', marker='x', linestyle='none', yerr=error, capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=3)

    # Plot the greyed out zone
    t_array = np.array(data['t'])
    T_array = np.array(data['T'])
    error_array = np.array(error)
    mask_grey = (t_array >= grey_lower) & (t_array <= grey_upper)
    plt.errorbar(t_array[mask_grey], T_array[mask_grey], color='#9e9e9e', marker='x', linestyle='none', yerr=error_array[mask_grey], capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=4)

    # Save the figure
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show() 

plot_data(data, error, ymin=18, ymax=25, xmax=180, xstep=20, xoffset=0, ystep=0.5, yoffset=-0, grey_lower=0, grey_upper=60, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/A3.pdf')