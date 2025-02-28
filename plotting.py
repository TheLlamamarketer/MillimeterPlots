import math
import numpy as np
import matplotlib.pyplot as plt
import random
import hashlib
import colorsys
import matplotlib.colors as mcolors

def generate_contrasting_color(index, total_colors, bg_hex='#FFFFFF', color_seed=100, 
                               initial_hue_offset=0.5, hue_range=0.7, saturation=0.7, value=0.75, variation=0.15):
    """
    Generate a contrasting color relative to a background color.
    If the background is too close to white or grayscale (low saturation),
    we just pick a random initial hue rather than relying on background hue.
    """
    random.seed(color_seed)
    
    bg_rgb = mcolors.hex2color(bg_hex)
    bg_hue, bg_sat, bg_val = colorsys.rgb_to_hsv(*bg_rgb)

    # If background saturation is too low, ignore its hue and pick a random initial hue
    if bg_sat < 0.05:  
        if index == 0:
            initial_hue = random.random()
            generate_contrasting_color._initial_hue = initial_hue
        else:
            initial_hue = getattr(generate_contrasting_color, '_initial_hue', 0.5)
        bg_hue = initial_hue

    hue_variation = random.uniform(-variation, variation)
    if total_colors > 1:
        hue = (bg_hue + initial_hue_offset + (index / (total_colors - 1)) * hue_range + hue_variation) % 1.0
    else:
        hue = (bg_hue + initial_hue_offset + hue_variation) % 1.0

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return mcolors.to_hex([*rgb])

def stable_hash(value):
    return int(hashlib.md5(value.encode()).hexdigest(), 16)

def varied_color(base_hex, color_seed=100, hue_range=0.04, index=0):
    random.seed(color_seed + stable_hash(base_hex) + index * 10)

    base_rgb = mcolors.to_rgb(base_hex)
    base_hue, base_sat, base_val = colorsys.rgb_to_hsv(*base_rgb)

    hue_offset = random.uniform(-hue_range, hue_range) if hue_range > 0 else 0.0
    new_hue = (base_hue + hue_offset) % 1.0

    rgb = colorsys.hsv_to_rgb(new_hue, base_sat, base_val)
    return mcolors.to_hex([*rgb])


def colors_from_groups(datasets, color_seed=None, bg_hex='#FFFFFF', hue_range=0.05):
    """
    Assign colors to datasets based on their color groups or individual IDs.
    If a dataset has a 'color' set, that color is used.
    Otherwise, new colors are generated to ensure contrast and variation.
    Datasets with the same 'color_group' will share a base color with slight variations.
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

    for group in color_group_dict:
        group_datasets = color_group_dict[group]

        specified_color = next((d['color'] for d in group_datasets if 'color' in d and d['color'] is not None), None)
        if specified_color:
            group_base_color = specified_color
        else:
            group_color_idx = color_indices[group]
            group_base_color = generate_contrasting_color(
                group_color_idx, total_colors_needed, bg_hex=bg_hex, color_seed=color_seed
            )
        
        for i, data in enumerate(group_datasets):
            if 'color' in data and data['color'] is not None:
                colors[id(data)] = data['color']
            else:
                colors[id(data)] = varied_color(group_base_color, color_seed=color_seed, hue_range=hue_range, index=i)

    for data in datasets_without_color_group:
        if 'color' in data and data['color'] is not None:
            colors[id(data)] = data['color']
        else:
            color_idx = color_indices[id(data)]
            colors[id(data)] = generate_contrasting_color(
                color_idx, total_colors_needed, bg_hex=bg_hex, color_seed=color_seed
            )
    return colors


def plot_grid(ax, width, height, bg_hex, xstep, xtickoffset, xmin, xmax, ystep, ytickoffset, ymin, ymax):

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

    ax.set_xticks(np.arange(xmin + xtickoffset, xmax + 1, xstep))
    ax.set_yticks(np.arange(ymin + ytickoffset, ymax + 1, ystep))

    # Draw grid lines
    tick_color = bg_hex
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

    return

def plot_color_seeds(seed_range=(100, 120), num_datasets=3, bg_hex='#FFFFFF', hue_range=0.05):
    """
    Visualize how different color seeds affect color assignment.
    - X-axis: Different `color_seed` values.
    - Y-axis: Different datasets (color groups).
    - Colors: Generated colors for each dataset with the given seed.
    """
    min_seed, max_seed = seed_range
    seeds = range(min_seed, max_seed + 1)
    
    # Generate dummy datasets (each assigned a color group)
    datasets = [{"color_group": f"group_{i}"} for i in range(num_datasets)]

    fig, ax = plt.subplots(figsize=(len(seeds) * 0.4, num_datasets * 0.8), constrained_layout=True)
    
    # Plot colors
    for col, seed in enumerate(seeds):
        # Generate colors for all datasets at once
        colors = colors_from_groups(datasets, color_seed=seed, bg_hex=bg_hex, hue_range=hue_range)

        for row, dataset in enumerate(datasets):
            color = colors[id(dataset)]  # Extract the color for the dataset
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color))

    # Formatting
    ax.set_xticks(np.arange(len(seeds)) + 0.5)
    ax.set_xticklabels(seeds, rotation=45, fontsize=10)
    ax.set_yticks(np.arange(num_datasets) + 0.5)
    ax.set_yticklabels([f"Dataset {i+1}" for i in range(num_datasets)], fontsize=10)

    ax.set_xlabel("Color Seed", fontsize=12)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_title("Color Variations Across Seeds", fontsize=14)

    ax.set_xlim(0, len(seeds))
    ax.set_ylim(0, num_datasets)
    ax.invert_yaxis()  # Align first dataset at the top
    ax.set_frame_on(False)
    
    plt.show()
    
def plot_data(filename, datasets, color_map=None, color_seed=203,
              title=None, xlabel=None, ylabel=None,
              legend_position='best', width=20, height=20,
              plot=True, ymax=None, ymin=None, xmax=None, xmin=None, 
              xticks=None, yticks=None, bg_hex='#FFFFFF'):
    
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54), constrained_layout=True)

    if color_map is None:
        color_map = {}
        for d in datasets:
            color_map[id(d)] = "#000000"

    if type(datasets) is not list:
        datasets = [datasets]

    colors = colors_from_groups(datasets, color_seed=color_seed, bg_hex=bg_hex)

    if xticks and yticks:
        plot_grid(ax, width, height, bg_hex, xticks[0], xticks[1], xmin, xmax, yticks[0], yticks[1], ymin, ymax)
    
    else:
        ax.grid(True, which='major', color='#CCCCCC', linestyle='-', linewidth=1)
        ax.grid(True, which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        ax.minorticks_on()

    for data in datasets:
        xdata = data.get('xdata', [])
        ydata = np.array(data.get('ydata', []), dtype=float)
        y_error = data.get('yerr')
        x_error = data.get('xerr')
        label = data.get('label')
        marker = data.get('marker', '.')
        line = data.get('line', 'None')
        confidence = data.get('confidence')
        confidence_label = data.get('confidence_label', True)
        fit = data.get('fit')
        fit_label = data.get('fit_label', True)
        fit_xdata = data.get('fit_xdata')
        fit_line = data.get('fit_line', '-')
        fit_error_lines = data.get('fit_error_lines')
        color = data.get('color', colors[id(data)] )

        if xdata is None or ydata is None or (not isinstance(xdata, (list, np.ndarray, float))) or (not isinstance(ydata, (list, np.ndarray, float))):
            continue

        if isinstance(xdata, float):
            xdata = [xdata]
        if isinstance(ydata, float):
            ydata = [ydata]
        
        if isinstance(xdata[0], str):
            xdata = np.array(range(len(xdata)))
            ax.set_xticks(xdata)
            ax.set_xticklabels(data.get('xdata', []))
        else:
            xdata = np.array(xdata, dtype=float)


        if confidence is not None:
            try:
                for i, (lower, upper) in enumerate(confidence):
                    if lower is not None and upper is not None:
                        ax.fill_between(
                            fit_xdata if fit_xdata is not None else xdata, lower, upper, interpolate=True,
                            facecolor=color, edgecolor=None, alpha=0.4 - (i * 0.1),
                            label=f"{label} CI ({i+1}σ)" if label and confidence_label else None, zorder=2
                        )
            except Exception as e:
                print(f"Error in confidence interval for dataset '{label}': {e}")
                continue

        if fit is not None:
            try:
                ax.plot(fit_xdata if fit_xdata is not None else xdata, fit, color=color, linestyle=fit_line, linewidth=1,
                        label=f"{label} Fit" if label and fit_label else None, zorder=3)
            except Exception as e:
                print(f"Error in fit for dataset '{label}': {e}")
                continue
        
        if fit_error_lines is not None:
            try:
                for (lower, upper) in fit_error_lines:
                    if lower is not None:
                        ax.plot(fit_xdata if fit_xdata is not None else xdata, lower, color=color, linestyle='--', linewidth=1,
                                label=f"{label} Fit Error" if label and fit_label else None, zorder=3)
                    if upper is not None:
                        ax.plot(fit_xdata if fit_xdata is not None else xdata, upper, color=color, linestyle='--', linewidth=1,
                                label=f"{label} Fit Error" if label and fit_label else None, zorder=3)
            except Exception as e:
                print(f"Error in fit error lines for dataset '{label}': {e}")
                continue


        

        if y_error is None and x_error is None:

            seen_points = {}
            for x, y in zip(xdata, ydata):
                point = (x, y)
                seen_points[point] = seen_points.get(point, 0) + 1

            xdata, ydata = zip(*seen_points.keys())

            log_counts = np.array([math.log(count + 1) for count in seen_points.values()])

            if log_counts.max() - log_counts.min() == 0:
                normalized = np.ones_like(log_counts)
            else:
                normalized = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min())

            sizes = 10 + normalized * (100 - 10)
            

            if marker != 'None':
                ax.scatter(xdata, ydata, color=color, marker=marker, s=sizes, alpha=0.9, zorder=4, label=label)
            if line != 'None':
                ax.plot(xdata, ydata, color=color, linestyle=line, linewidth=1, zorder=4, label=label, marker=marker, markersize=10)
        else:
            try:
                if isinstance(y_error, tuple) and len(y_error) == 2:
                    yerr_lower, yerr_upper = y_error
                    if isinstance(yerr_lower, float):
                        yerr_lower = np.full_like(ydata, yerr_lower)
                        yerr_upper = np.full_like(ydata, yerr_upper)
                elif isinstance(y_error, float):
                    yerr_lower = np.full_like(ydata, y_error)
                    yerr_upper = np.full_like(ydata, y_error)
                else:
                    yerr_lower = y_error
                    yerr_upper = y_error


                if isinstance(x_error, tuple) and len(x_error) == 2:
                    xerr_lower, xerr_upper = x_error
                elif isinstance(x_error, float):
                    xerr_lower = np.full_like(xdata, x_error)
                    xerr_upper = np.full_like(xdata, x_error)
                else:
                    xerr_lower = x_error
                    xerr_upper = x_error


                ax.errorbar(
                    xdata, ydata, yerr=[np.abs(yerr_lower), np.abs(yerr_upper)] if y_error is not None else None,
                    xerr=[np.abs(xerr_lower), np.abs(xerr_upper)] if x_error is not None else None,
                    fmt=marker, color=color, linestyle='none',
                    capsize=3, elinewidth=0.6, capthick=0.6,
                    label=label, markersize=8, alpha=0.9, zorder=4
                )
                
            except Exception as e:
                print(f"Error in error bars for dataset '{label}': {e}")
                continue
        
    handles, labels = ax.get_legend_handles_labels()
    if any(labels) and legend_position:
        ax.legend(loc=legend_position)

    ax.set_xlabel(xlabel if xlabel else '')
    ax.set_ylabel(ylabel if ylabel else '')
    ax.set_title(title if title else '')

    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlim(left=xmin, right=xmax)

    # Save the figure including all elements
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    if plot:
        plt.show()
    else:
        plt.close()
