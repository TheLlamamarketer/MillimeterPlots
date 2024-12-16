import numpy as np
import matplotlib.pyplot as plt
import random
import colorsys
import matplotlib.colors as mcolors

def generate_contrasting_color(index, total, bg_hex='#FFFFFF', color_seed=None, 
                               base_hue_offset=0.5, saturation=0.7, value=0.6):
    """
    Generate a contrasting color relative to a (possibly neutral) background color.
    If the background is too close to white or grayscale (low saturation),
    we just pick a random initial hue rather than relying on background hue.
    """
    # Seed once outside this function if needed, or just here if you want
    if color_seed is not None and index == 0:
        random.seed(color_seed)
    
    bg_rgb = mcolors.hex2color(bg_hex)
    bg_hue, bg_sat, bg_val = colorsys.rgb_to_hsv(*bg_rgb)

    # If background saturation is too low, ignore its hue and pick a random initial hue
    if bg_sat < 0.05:  
        if index == 0:
            # Random initial hue if this is the first dataset
            initial_hue = random.random() if color_seed is not None else 0.0
            # Store it as a global or static variable if you need consistency
            generate_contrasting_color._initial_hue = initial_hue
        else:
            initial_hue = getattr(generate_contrasting_color, '_initial_hue', 0.5)
        bg_hue = initial_hue

    # Slight variation in hue if total > 1
    if total > 1:
        # Introduce some hue variation per dataset
        hue_variation = random.uniform(-0.1, 0.1) if color_seed is not None else 0.0
        hue = (bg_hue + base_hue_offset + (index / (total - 1)) * 0.5 + hue_variation) % 1.0
    else:
        # Single dataset, no scaling by total
        hue_variation = random.uniform(-0.1, 0.1) if color_seed is not None else 0.0
        hue = (bg_hue + base_hue_offset + hue_variation) % 1.0

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

def colors_from_groups(datasets, color_seed=None, bg_hex='#FFFFFF'):
    """
    Assign colors to datasets based on their color groups or individual IDs.
    If a dataset has a 'color' set, that color is used.
    Otherwise, colors are generated to maximize contrast from a background color.
    """
    color_group_dict = {}
    datasets_without_color_group = []

    # Separate datasets by color_group
    for data in datasets:
        color_group = data.get('color_group')
        if color_group is not None:
            color_group_dict.setdefault(color_group, []).append(data)
        else:
            datasets_without_color_group.append(data)

    total_colors_needed = len(color_group_dict) + len(datasets_without_color_group)

    # Assign color indices to each group or dataset without a group
    color_indices = {}
    idx = 0
    for group in color_group_dict:
        color_indices[group] = idx
        idx += 1
    for data in datasets_without_color_group:
        # Use object's ID as a unique key
        color_indices[id(data)] = idx
        idx += 1

    # Generate or retrieve colors
    colors = {}
    for key, color_idx in color_indices.items():
        if isinstance(key, str):
            # This is a color_group
            datasets_in_group = color_group_dict[key]
            # If any dataset in group specifies a color, use that
            specified_color = next((d['color'] for d in datasets_in_group if 'color' in d and d['color'] is not None), None)
            if specified_color:
                colors[key] = specified_color
            else:
                # Generate a color for the group
                colors[key] = generate_contrasting_color(
                    color_idx, total_colors_needed, bg_hex=bg_hex, color_seed=color_seed
                )
        else:
            # This is a dataset without a group
            data = next(d for d in datasets_without_color_group if id(d) == key)
            if 'color' in data and data['color'] is not None:
                colors[key] = data['color']
            else:
                colors[key] = generate_contrasting_color(
                    color_idx, total_colors_needed, bg_hex=bg_hex, color_seed=color_seed
                )
    return colors



def plot_data(filename, datasets, title=None, x_label=None, y_label=None,
              legend_position='best', color_seed=203, width=20, height=28,
              plot=True, ymax=None, ymin=None, xmax=None, xmin=None, bg_hex='#FFFFFF'):
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54))

    # Generate colors for datasets based on color groups or unique identifiers 
    colors = colors_from_groups(datasets, color_seed=color_seed, bg_hex=bg_hex)

    ax.grid(True, which='major', color='#CCCCCC', linestyle='-', linewidth=1)
    ax.grid(True, which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    # Plot each dataset
    for data in datasets:
        xdata = np.array(data.get('xdata', []), dtype=float)
        ydata = np.array(data.get('ydata', []), dtype=float)

        y_error = data.get('y_error')
        x_error = data.get('x_error')
        label = data.get('label')
        marker = data.get('marker', 'x')
        line = data.get('line', '-')
        line_fit = data.get('line_fit', '-')
        confidence = data.get('confidence', {})
        fit = data.get('fit', None)
        color_group = data.get('color_group')
        high_res_x = data.get('high_res_x', None)

        color = data.get('color')
        if color is None:
            color = colors[color_group if color_group else id(data)]

        if xdata.size == 0 or ydata.size == 0:
            continue

        if confidence:
            for i, (lower, upper) in enumerate(confidence):
                if lower is not None and upper is not None:
                    ax.fill_between(
                        high_res_x if high_res_x is not None else xdata, lower, upper, interpolate=True,
                        facecolor=color, edgecolor=None, alpha=0.4 - (i * 0.1),
                        label=f"{label} CI ({i+1}σ)" if label else None, zorder=2
                    )

        if fit is not None:
            try:
                ax.plot(high_res_x if high_res_x is not None else xdata, fit, color=color, linestyle=line_fit, linewidth=1,
                        label=f"{label} Fit" if label else None, zorder=5)
            except Exception as e:
                print(f"Error in fit for dataset '{label}': {e}")
                continue

        if y_error is None and x_error is None:
            ax.plot(xdata, ydata, color=color, marker=marker, linestyle=line,
                    label=label, markersize=10, alpha=0.9, zorder=4)
        else:
            if isinstance(y_error, tuple) and len(y_error) == 2:
                yerr_lower, yerr_upper = y_error
            else:
                yerr_lower = y_error
                yerr_upper = y_error

            if isinstance(x_error, tuple) and len(x_error) == 2:
                xerr_lower, xerr_upper = x_error
            else:
                xerr_lower = x_error
                xerr_upper = x_error

            ax.errorbar(
                xdata, ydata, yerr=[np.abs(yerr_lower), np.abs(yerr_upper)] if y_error is not None else None,
                xerr=[np.abs(xerr_lower), np.abs(xerr_upper)] if x_error is not None else None,
                fmt=marker, color=color, linestyle='none',
                capsize=2, elinewidth=0.5, capthick=0.5,
                label=label, markersize=5, alpha=0.9, zorder=4
            )
       
    handles, labels = ax.get_legend_handles_labels()
    if any(labels) and legend_position:
        ax.legend(loc=legend_position)

    ax.set_xlabel(x_label if x_label else '')
    ax.set_ylabel(y_label if y_label else '')
    ax.set_title(title if title else '')

    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(left=xmin, right=xmax)

    plt.tight_layout()
    plt.savefig(filename, format='pdf')

    if plot:
        plt.show()
    else:
        plt.close()
