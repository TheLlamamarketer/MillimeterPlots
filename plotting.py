import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors
import random

def generate_contrasting_color(index, total, bg_hex='#eb3107', seed=None):
    if seed is not None:
        random.seed(seed + index)
    # Convert background color to HSV
    bg_rgb = mcolors.hex2color(bg_hex)
    bg_hsv = colorsys.rgb_to_hsv(*bg_rgb)
    bg_hue = bg_hsv[0]

    # Introduce a small random variation in hue
    hue_variation = random.uniform(-0.1, 0.1) if seed is not None else 0
    hue = (bg_hue + 0.5 + (index / max(1, total - 1)) * 0.5 + hue_variation) % 1.0 if total > 1 else (bg_hue + 0.5) % 1.0
    saturation = 0.7
    value = 0.6

    # Convert HSV back to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    # Convert RGB to hex
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

colors= ['#eb3107', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# '#eb3107'
def plot_data(filename, datasets, ymin, ymax, ystep, xmax, xmin, xstep, ytickoffset=0, xtickoffset=0, grey_zones=None, width=20, height=28, background_color='#eb3107', x_label=None, y_label=None, legend_position='best'):
    fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54))
    aspect_ratio = (xmax - xmin) / (ymax - ymin) * (height / width)
    ax.set_aspect(aspect_ratio, adjustable='box')

    # Generate grid lines
    factors = [1, 2, 10]
    main_lines_x, secondary_lines_x, tertiary_lines_x = [
        np.arange(xmin, xmax, (xmax - xmin) / (width * factor)) for factor in factors
    ]
    main_lines_y, secondary_lines_y, tertiary_lines_y = [
        np.arange(ymin, ymax, (ymax - ymin) / (height * factor)) for factor in factors
    ]

    # Set ticks
    ax.set_xticks(np.arange(xmin + xtickoffset, xmax + 1, xstep))
    ax.set_yticks(np.arange(ymin + ytickoffset, ymax + 1, ystep))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Draw grid lines
    tick_color = background_color
    for lines_x, lines_y, line_width in [[tertiary_lines_x, tertiary_lines_y, 0.1],
                                         [secondary_lines_x, secondary_lines_y, 0.45],
                                         [main_lines_x, main_lines_y, 0.7]]:
        for tick in lines_x:
            ax.axvline(x=tick, color=tick_color, linestyle='-', linewidth=line_width, zorder=0)
        for tick in lines_y:
            ax.axhline(y=tick, color=tick_color, linestyle='-', linewidth=line_width, zorder=0)

    # Draw border lines
    for spine in ax.spines.values():
        spine.set_edgecolor(tick_color)
        spine.set_zorder(2)

    # Plot each dataset
    for idx, data in enumerate(datasets):
        xdata = np.array(data['xdata'], dtype=float)
        ydata = np.array(data['ydata'], dtype=float)
        y_error = np.array(data['y_error'], dtype=float) if 'y_error' in data and data['y_error'] is not None else None
        x_error = np.array(data['x_error'], dtype=float) if 'x_error' in data and data['x_error'] is not None else None
        color = data.get('color')
        label = data.get('label')

        # If no color is provided, generate a random one if there are multiple datasets, otherwise use black
        if len(datasets) > 1 and color is None:
            color = generate_contrasting_color(idx, len(datasets), seed=105, bg_hex=background_color)
        elif color is None:
            color = "black"

        ax.errorbar(xdata, ydata, color=color, marker='x', linestyle='none', yerr=y_error, xerr=x_error, capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=3, label=label)

        # Adding grey zones to show the excluded areas of the fit
        if grey_zones:
            for zone in grey_zones:
                mask_grey = np.ones_like(xdata, dtype=bool)

                if 'x_low' in zone and 'x_up' in zone:
                    mask_grey &= (xdata >= zone['x_low']) & (xdata <= zone['x_up'])
                if 'y_low' in zone and 'y_up' in zone:
                    mask_grey &= (ydata >= zone['y_low']) & (ydata <= zone['y_up'])
                if 'x_val' in zone:
                    mask_grey &= (xdata == zone['x_val'])
                if 'y_val' in zone:
                    mask_grey &= (ydata == zone['y_val'])
                if np.any(mask_grey):
                    ax.errorbar(xdata[mask_grey], ydata[mask_grey],
                        color='#9e9e9e', marker='x', linestyle='none',
                        yerr=y_error[mask_grey] if y_error is not None else None,
                        xerr=x_error[mask_grey] if x_error is not None else None,
                        capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=4
                    )

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
