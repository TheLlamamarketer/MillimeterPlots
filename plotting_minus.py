from calendar import c
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

def colors_from_groups(datasets, color_seed=None):
    color_group_dict = {}
    datasets_without_color_group = []

    # Separate datasets with and without color groups
    for data in datasets:
        color_group = data.get('color_group')
        if color_group is not None:
            if color_group not in color_group_dict:
                color_group_dict[color_group] = []
            color_group_dict[color_group].append(data)
        else:
            datasets_without_color_group.append(data)

    total_colors_needed = len(color_group_dict) + len(datasets_without_color_group)

    # Assign color indices
    color_indices = {}
    idx = 0
    for color_group in color_group_dict.keys():
        color_indices[color_group] = idx
        idx += 1
    for data in datasets_without_color_group:
        color_indices[id(data)] = idx  # Use id(data) as unique identifier
        idx += 1

    # Generate colors
    colors = {}
    for key, color_idx in color_indices.items():
        # For color groups
        if isinstance(key, str):
            color_group = key
            datasets_in_group = color_group_dict[color_group]
            color_specified = any('color' in d and d['color'] is not None for d in datasets_in_group)
            if color_specified:
                specified_color = next(d['color'] for d in datasets_in_group if 'color' in d and d['color'] is not None)
                colors[color_group] = specified_color
            else:
                color = generate_contrasting_color(color_idx, total_colors_needed, color_seed=color_seed)
                colors[color_group] = color
        # For datasets without color groups
        else:
            data_id = key
            data = next(d for d in datasets_without_color_group if id(d) == data_id)
            if 'color' in data and data['color'] is not None:
                colors[data_id] = data['color']
            else:
                color = generate_contrasting_color(color_idx, total_colors_needed, color_seed=color_seed)
                colors[data_id] = color
    return colors



def plot_data(filename, datasets, x_label=None, y_label=None, legend_position='best', color_seed=203, width=20, height=28, plot=True):
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54))

    # Generate colors for datasets based on color groups or unique identifiers 
    colors = colors_from_groups(datasets, color_seed=color_seed)

    # Plot each dataset
    for idx, data in enumerate(datasets):
        xdata = np.array(data['xdata'], dtype=float) if 'xdata' in data and data['xdata'] is not None else None
        ydata = np.array(data['ydata'], dtype=float) if 'ydata' in data and data['ydata'] is not None else None
        y_error = np.array(data['y_error'], dtype=float) if 'y_error' in data and data['y_error'] is not None else None
        x_error = np.array(data['x_error'], dtype=float) if 'x_error' in data and data['x_error'] is not None else None
        color = data.get('color')
        label = data.get('label')
        marker = data.get('marker', 'x')
        line = data.get('line', '-')
        confidence = data.get('confidence', False)
        low_bound = data.get('low_bound')
        high_bound = data.get('high_bound')
        color_group = data.get('color_group')

        if color is None:
            if color_group is not None:
                color = colors[color_group]
            else:
                color = colors[id(data)]

        ax.grid(True, which='major')
        ax.grid(True, which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax.minorticks_on()


        if confidence:
            ax.fill_between(xdata, low_bound, high_bound, color=color, alpha=0.5, label=label, clip_on=False, zorder=2 )
        elif y_error is None and x_error is None:
            ax.plot(xdata, ydata, color=color, marker=marker, linestyle=line, clip_on=False, label=label,  markersize=10)
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

    if plot:
        plt.show()
    else:
        plt.close()
