
# Documentation for the `plot_data` and `find_clean_intervals` Functions

## 1. `plot_data` Function

### Parameter Descriptions

1. **filename (str)**:
    - The path where the plot will be saved as a PDF file.

2. **datasets (list of dicts)**:
    - Each dictionary represents a dataset with the following keys:
        - `'xdata'` (list or array): X-axis values.
        - `'ydata'` (list or array): Y-axis values.
        - `'x_error'` (list or array, optional): Errors for X values.
        - `'y_error'` (list or array, optional): Errors for Y values.
        - `'color'` (str, optional): Color for the dataset.
        - `'label'` (str, optional): Label for the legend.

3. **ymin, ymax (float)**:
    - Minimum and maximum values for the Y-axis.

4. **ystep (float)**:
    - Step size for Y-axis ticks.

5. **xmin, xmax (float)**:
    - Minimum and maximum values for the X-axis.

6. **xstep (float)**:
    - Step size for X-axis ticks.

7. **ytickoffset, xtickoffset (float, optional)**:
    - Offset values for the Y-axis and X-axis ticks, incase the axis starts off on an unclean value, like 37, instead of 40.

8. **grey_zones (list of dicts, optional)**:
    - List of dictionaries defining grey zones on the plot.
    - Possible keys in each dictionary:
        - `'x_low'` (float): Lower bound of the grey zone on the X-axis.
        - `'x_up'` (float): Upper bound of the grey zone on the X-axis.
        - `'y_low'` (float): Lower bound of the grey zone on the Y-axis.
        - `'y_up'` (float): Upper bound of the grey zone on the Y-axis.
    -These two can be used together or seperately:
        - `'x_val'` (float): Specific X value to include in the grey zone.
        - `'y_val'` (float): Specific Y value to include in the grey zone.

9. **width, height (float, optional)**:
    - Width and height of the plot in cm.
    - Automatically set to 28 cm width and 20cm height

10. **background_color (str, optional)**:
    - Hex code for the background color of the plot.
    - Automatically selects the red-orange color of real millimeter paper sheets.

11. **x_label, y_label (str, optional)**:
    - Labels for the X and Y axes.

12. **legend_position (str, optional)**:
    - Position of the legend (e.g., 'best', 'upper right').

### Function Workflow

1. **Setting Up the Plot**: Initializes a plot with specified dimensions and aspect ratio.
2. **Drawing Grid Lines**: Creates three levels of grid lines (main, secondary, and tertiary) based on the figure size.
3. **Plotting Data**: Each dataset is plotted with error bars. If no color is provided, it generates contrasting colors.
4. **Highlighting Grey Zones**: Plots specific regions of interest as grey zones to show excluded areas or additional context.
5. **Configuring Legend and Labels**: Adds legend and labels if provided.
6. **Saving and Displaying the Plot**: Saves the plot to the specified file and displays it.

---

## 2. `find_interval` Functions

def find_interval(xdata, ydata, name=' ', xerror=None, yerror=None, width=20, height=28, max_increment_height=1, max_increment_width=1)

This function calculates clean intervals for the width and height of the plot, given the data and their associated errors.

### Parameter Descriptions for `plot_data`

1. **xdata, ydata (array)**:
    - The data for which intervals need to be calculated.

2. **name (str)**:
    - Name of the data, used for display purposes.

3. **xerror, yerror (array, optional)**:
    - Optional errors associated with X and Y data.

4. **width, height (float)**:
    - Width and height of the plot.

5. **max_increment_height, max_increment_width (float)**:
    - Maximum increments allowed for height and width calculations.
