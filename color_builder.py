import random
import colorsys
import matplotlib.colors as mcolors
from plotting import generate_contrasting_color, varied_color

def build_color_map(all_datasets, color_seed=100, bg_hex='#FFFFFF', hue_range=0.05):
    """
    Build a single color map for all datasets at once.
    - Datasets that share 'color_group' get the same base color (slight variations).
    - Datasets with a pre-specified 'color' keep it.
    - Ungrouped datasets each get their own color.
    Returns: a dict {id(dataset): "#hexcolor"}.
    """
    # Group datasets by color_group
    color_group_dict = {}
    ungrouped = []
    for data in all_datasets:
        grp = data.get('color_group')
        if grp is not None:
            color_group_dict.setdefault(grp, []).append(data)
        else:
            ungrouped.append(data)

    # We'll assign each group (and each ungrouped dataset) a base index
    all_groups = sorted(color_group_dict.keys())
    total_needed = len(all_groups) + len(ungrouped)

    color_indices = {}
    idx = 0
    for grp in all_groups:
        color_indices[grp] = idx
        idx += 1
    for data in ungrouped:
        color_indices[id(data)] = idx
        idx += 1

    # Build the map of dataset-id -> color
    color_map = {}

    # 1) Assign colors for grouped datasets
    for grp in all_groups:
        grp_datasets = color_group_dict[grp]
        # If any dataset in the group has an explicit 'color', use that as the base
        specified_color = next((d['color'] for d in grp_datasets if d.get('color')), None)
        if specified_color:
            base_color = specified_color
        else:
            # generate a base color for the group
            grp_index = color_indices[grp]
            base_color = generate_contrasting_color(
                index=grp_index,
                total_colors=total_needed,
                bg_hex=bg_hex,
                color_seed=color_seed
            )
        # Now assign each dataset in the group a slight variation unless it has its own color
        for i, ds in enumerate(grp_datasets):
            if ds.get('color'):
                color_map[id(ds)] = ds['color']
            else:
                color_map[id(ds)] = varied_color(
                    base_color, color_seed=color_seed, hue_range=hue_range, index=i
                )

    # 2) Assign colors for ungrouped datasets
    for data in ungrouped:
        if data.get('color'):
            color_map[id(data)] = data['color']
        else:
            base_index = color_indices[id(data)]
            color_map[id(data)] = generate_contrasting_color(
                index=base_index,
                total_colors=total_needed,
                bg_hex=bg_hex,
                color_seed=color_seed
            )

    return color_map