import numpy as np
import matplotlib.pyplot as plt

data = {
    't': list(range(21)),
    'T': [19.9, 20.6, 21.2, 21.9, 22.6, 23.3, 24, 24.6, 25.3, 25.8, 25.8, 25.9, 26.3, 26.7, 27.2, 27.7, 28.2, 28.7, 29.3, 29.8, 30.4]
}
error = [i * 0.001 + 0.2 for i in data['T']]

def plot_data_horizontal(data, error, xmin, xmax, ymin, ymax, xstep, xoffset, ystep, yoffset, grey_lower, grey_upper, filename):
    num_main_lines_x = 28
    num_main_lines_y = 18

    fig, ax = plt.subplots(figsize=(num_main_lines_x / 2.54, num_main_lines_y / 2.54))

    main_lines_x = np.arange(xmin, xmax, (xmax - xmin) / num_main_lines_x)
    main_lines_y = np.arange(ymin, ymax, (ymax - ymin) / num_main_lines_y)

    secondary_lines_x = np.arange(xmin, xmax, (xmax - xmin) / (num_main_lines_x * 2))
    secondary_lines_y = np.arange(ymin, ymax, (ymax - ymin) / (num_main_lines_y * 2))

    tertiary_lines_x = np.arange(xmin, xmax, (xmax - xmin) / (num_main_lines_x * 10))
    tertiary_lines_y = np.arange(ymin, ymax, (ymax - ymin) / (num_main_lines_y * 10))

    ax.set_xticks(np.arange(xmin + xoffset, xmax + 0.1, xstep))
    ax.set_yticks(np.arange(ymin, ymax + 0.1, ystep))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Tertiary grid lines
    for tick in tertiary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.1)
    for tick in tertiary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.1)

    # Secondary grid lines
    for tick in secondary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.45)
    for tick in secondary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.45)

    # Main grid lines
    for tick in main_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.7)
    for tick in main_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_edgecolor('#eb3107')
        spine.set_zorder(1)


    filtered_t = []
    filtered_T = []
    for t, T in zip(data['t'], data['T']):
        if (T <= xmax) and (t <= ymax):
            filtered_t.append(t)
            filtered_T.append(T)
    
    plt.errorbar(filtered_T, filtered_t, color='black', marker='x', linestyle='none', xerr=[i * 0.001 + 0.2 for i in filtered_T], capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=3)
    
    slope, intercept = np.polyfit(filtered_T, filtered_t, 1)
    regression_line = slope * np.array(data['T']) + intercept
    #plt.plot(data['T'], regression_line, color='blue')

    print(slope, intercept)

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()
 

plot_data_horizontal(data, error, xmin=19.4, xmax=30.6, ymin=0, ymax=20, xstep=1, xoffset=0.6, ystep=1, yoffset=0, grey_lower=0, grey_upper=0, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/A1.pdf')

plot_data_horizontal(data, error, xmin=19.5, xmax=25.9, ymin=0, ymax=8, xstep=0.8, xoffset=0.0, ystep=1, yoffset=0, grey_lower=0, grey_upper=0, filename='C:/Users/alexa/OneDrive/Desktop/newfolder/A1-2.pdf')