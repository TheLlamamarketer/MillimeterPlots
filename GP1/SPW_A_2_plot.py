import numpy as np
import matplotlib.pyplot as plt

data_fe = {
    't': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    'T': [3.5, 5.4, 6.9, 7.4, 8.1, 8.3, 8.3, 8.4, 8.5, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6]
}

data_al = {
    't': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600],
    'T': [3.2, 4.4, 5.1, 5.4, 5.6, 5.8, 5.8, 5.9, 5.9, 6, 6, 6.1, 6.1, 6.1, 6.2, 6.2, 6.2, 6.3, 6.3, 6.3, 6.4, 6.4, 6.4, 6.5, 6.5, 6.5, 6.5, 6.6, 6.6, 6.6, 6.7, 6.7, 6.7, 6.8, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 7, 7, 7, 7, 7.1, 7.1, 7.1, 7.1, 7.2, 7.2, 7.2, 7.2, 7.3, 7.3, 7.3, 7.3, 7.4, 7.4, 7.4, 7.5]
}

error_fe = [i * 0.001 + 0.2 for i in data_fe['T']]
error_al = [i * 0.001 + 0.2 for i in data_al['T']]

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

plot_data(data_fe, error_fe, ymin=3.3, ymax=8.9, xmax=150, xstep=10, xoffset=0, ystep=0.5, yoffset=-0.3, grey_lower=0, grey_upper=50, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/A2-Fe.pdf')

# Plot and save the figure for Al data
plot_data(data_al, error_al, ymin=2.6, ymax=8.2, xmax=600, xstep=40, xoffset=0, ystep=0.5, yoffset=-1.6, grey_lower=0, grey_upper=50, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/A2-Al.pdf')
