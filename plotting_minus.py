import numpy as np
import matplotlib.pyplot as plt
import random
import colorsys
import matplotlib.colors as mcolors

def generate_contrasting_color(index, total, bg_hex='#FFFFFF', color_seed=None):
    if color_seed is not None:
        random.seed(color_seed + index)
    # Convert background color to HSV
    bg_rgb = mcolors.hex2color(bg_hex)
    bg_hsv = colorsys.rgb_to_hsv(*bg_rgb)
    bg_hue = bg_hsv[0]

    # Generate hue based on index for contrast
    hue_variation = random.uniform(-0.1, 0.1) if color_seed is not None else 0
    hue = (bg_hue + 0.5 + (index / max(1, total - 1)) * 0.5 + hue_variation) % 1.0 if total > 1 else (bg_hue + 0.5) % 1.0
    saturation = 0.7
    value = 0.6

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

def plot_data(filename, datasets, x_label=None, y_label=None, legend_position='best', color_seed=203, width=20, height=28):
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54))
 
    # Plot each dataset
    for idx, data in enumerate(datasets):
        xdata = np.array(data['xdata'], dtype=float)
        ydata = np.array(data['ydata'], dtype=float)
        y_error = np.array(data['y_error'], dtype=float) if 'y_error' in data and data['y_error'] is not None else None
        x_error = np.array(data['x_error'], dtype=float) if 'x_error' in data and data['x_error'] is not None else None
        color = data.get('color')
        label = data.get('label')
        marker = data.get('marker', 'x')
        line = data.get('line', '-')

        # Generate color if not specified
        if len(datasets) > 1 and color is None:
            color = generate_contrasting_color(idx, len(datasets), color_seed=color_seed)
        elif color is None:
            color = "black"

        ax.grid(True)

        # Plot data points with error bars
        if y_error is None and x_error is None:
            ax.plot(xdata, ydata, color=color, marker=marker, linestyle=line, clip_on=False, label=label)

        else:
            ax.errorbar(xdata, ydata, color=color, marker=marker, linestyle='none', 
                    yerr=y_error, xerr=x_error, capsize=4, elinewidth=1, 
                    capthick=1, clip_on=False, label=label, markersize=10)

    # Add legend if labels are provided
    handles, labels = ax.get_legend_handles_labels()
    if any(labels) and legend_position:
        ax.legend(loc=legend_position)

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()
