from help import round_val
import numpy as np

def print_standard_table(
    data,
    headers,  
    header_groups=None,
    caption=None,
    label=None,
    column_formats=None,
    si_setup=None,
    show=True
):
    """
    Prints a standard LaTeX table with optional group headers, specified precision, and column-specific formats.

    The headers dictionary affects the table as follows:
    - "label": The column header label in the LaTeX table.
    - "err": The error values associated with the data, used for displaying uncertainties.
    - "data": The actual data to be displayed in the column. If not provided, data is taken from the main data dictionary.
    - "dark": If True, the column will have a dark background.
    - "intermed": If True, intermediate rounding is applied to the values.
    - "repeat": If False, a single value will only be displayed in the first row.

    Parameters:
    - data (dict): Dictionary containing the data for the table.
    - headers (dict): Dictionary mapping column keys to a dictionary with "label", "err", "data", etc.
    - header_groups (list of tuples): Optional group headers in the form (group_name, number_of_columns).
    - caption (str): The caption for the table.
    - label (str): The label for referencing the table in LaTeX.
    - column_formats (list): List of specific `table-format` strings for each column.
    - si_setup (str): The siunitx package settings for numerical formatting for the entire table if no specific format is provided.
    - show (bool): Whether to display the table output.
    """

    if not show:
        return

    num_columns = len(headers)

    # Get the column keys in order
    column_keys = list(headers.keys())

    layout_parts = []
    if column_formats:
        for key, fmt in zip(column_keys, column_formats):
            entry = headers[key]
            dark = entry.get("dark", False)
            col_format = f"S[table-format={fmt}]"
            if dark:
                col_format = f">{{\\columncolor{{black!20}}}}{col_format}"
            layout_parts.append(col_format)
    else:
        for key in column_keys:
            entry = headers[key]
            dark = entry.get("dark", False)
            if si_setup:
                col_format = f"S[{si_setup}]"
            else:
                col_format = "c"
            if dark:
                col_format = f">{{\\columncolor{{black!20}}}}{col_format}"
            layout_parts.append(col_format)
    layout = "|".join(layout_parts)
    layout = f"| {layout} |"

    # Print LaTeX table preamble
    print("\\begin{table}[h!]")
    print("    \\centering")
    if caption:
        print(f"    \\caption{{{caption}}}")
    if si_setup and not column_formats:
        print(f"    \\sisetup{{{si_setup}}}")  # Apply si_setup globally only if no column-specific format is provided
    print(f"    \\begin{{tabular}}{{{layout}}}")
    print("    \\toprule")

    # Print group headers if provided
    if header_groups:
        group_row = " & ".join(
            [f"\\multicolumn{{{span}}}{{|c|}}{{{name}}}" for name, span in header_groups]
        )
        print(f"    {group_row} \\\\")
        print("    \\midrule")

    # Print main headers derived from headers values (using label)
    header_row = " & ".join(entry["label"] for entry in headers.values()) + " \\\\"
    print(f"    {header_row}")
    print("    \\midrule")

    # Determine the maximum number of rows
    max_rows = 0
    for key, entry in headers.items():
        # Check if data is provided in entry or in data dict
        if 'data' in entry:
            column_data = entry['data']
            if isinstance(column_data, (list, np.ndarray)):
                length = len(column_data)
            else:
                length = 1  # Single value
        else:
            column_data = data.get(key, [])
            if isinstance(column_data, (list, np.ndarray)):
                length = len(column_data)
            else:
                length = 1  # Single value
        if length > max_rows:
            max_rows = length

    # Print data rows
    for row in range(max_rows):
        row_data = []
        for key, entry in headers.items():
            try:
                # Get data for this column
                if 'data' in entry:
                    column_data = entry['data']
                else:
                    column_data = data.get(key, [])
                # Determine the value
                if isinstance(column_data, (list, np.ndarray)):
                    if row < len(column_data):
                        value = column_data[row]
                    else:
                        value = None
                else:
                    # Single value
                    repeat = entry.get('repeat', True)
                    if repeat:
                        value = column_data
                    else:
                        value = column_data if row == 0 else None

                # Handle None values
                if value is None:
                    formatted_value = ""
                elif isinstance(value, str):
                    formatted_value = f"{{{value}}}"  # Plain string with no error
                else:
                    # Now handle errors
                    if "err" in entry and value != "":
                        error = entry["err"]
                        if isinstance(error, (list, np.ndarray)):
                            if row < len(error):
                                error_value = error[row]
                            else:
                                error_value = 0
                        elif isinstance(error, dict):
                            error_value = error.get(row, 0)
                        elif isinstance(error, (float, int)):
                            error_value = error
                        else:
                            error_value = 0
                    else:
                        error_value = 0

                    # Decide on intermed condition
                    intermed = entry.get("intermed", False)
                    round_value = entry.get("round", True)

                    if not round_value:
                        if error_value != 0:
                            formatted_value = "{$" + f"{value}" + " \\pm " + f"{error_value}" + "$}"
                        else:
                            formatted_value = "{$" + f"{value}" + "$}"
                    else:
                        # Handle cases where value is zero
                        if value == 0 and error_value != 0:
                            formatted_value = "{$" + f"{value}" + " \\pm " + f"{error_value}" + "$}"
                        elif error_value != 0:
                            rounded_val, err_round, power = round_val(value, err=error_value, intermed=intermed)

                            if power <= 0:
                                formatted_val_str = f"{rounded_val:.0f}"
                                formatted_err_str = f"{err_round:.0f}"
                            else:
                                formatted_val_str = f"{rounded_val:.{power}f}"
                                formatted_err_str = f"{err_round:.{power}f}"

                            formatted_value = "{$" + f"{formatted_val_str}" + " \\pm " + f"{formatted_err_str}" + "$}"
                        else:
                            rounded_val, power = round_val(value, err=0, intermed=intermed)

                            if power <= 0:
                                formatted_val_str = f"{rounded_val:.0f}"
                            else:
                                formatted_val_str = f"{rounded_val:.{power}f}"

                            formatted_value = "{$" + f"{formatted_val_str}" + "$}"

            except (IndexError, KeyError) as e:
                formatted_value = ""
                print(f"Error processing row {row}, column {key}: {e}")
            except Exception as e:
                formatted_value = ""
                print(f"Unexpected error processing row {row}, column {key}: {e}")

            row_data.append(formatted_value)

        # Create row string and print
        row_str = " & ".join(row_data) + " \\\\"
        print(f"    {row_str}")

    # Print table closing elements
    print("    \\bottomrule")
    print("    \\end{tabular}")
    if label:
        print(f"    \\label{{{label}}}")
    print("\\end{table}")

    return



def print_complex_table( data, headers, header_groups=None, caption=None, label=None, column_formats=None, si_setup=None, show=True):
    """
    Parameters:
    data (dict or list of dicts): The data to be displayed in the table. Each dictionary represents a block of data.
    headers (dict): A dictionary where keys are column identifiers and values are dictionaries containing header information.
        Each header dictionary should have a "label" key for the column label and optionally "dark" for dark column formatting.
    header_groups (list of tuples, optional): A list of tuples where each tuple contains a group name and the span of columns it covers.
    caption (str, optional): The caption for the table.
    label (str, optional): The label for the table, used for referencing in LaTeX.
    column_formats (list of str, optional): A list of column formats for each column. If not provided, default formats are used.
    si_setup (str, optional): SI unit setup string for formatting numerical values.
    show (bool, optional): If False, the function will not print the table. Default is True.
    Returns:
    None
    """

    if not show:
        return

    # Get the column keys in order
    column_keys = list(headers.keys())

    layout_parts = []
    if column_formats:
        for key, fmt in zip(column_keys, column_formats):
            entry = headers[key]
            dark = entry.get("dark", False)
            col_format = fmt if fmt == 'l' else f"S[table-format={fmt}]"
            if dark:
                col_format = f">{{\\columncolor{{black!20}}}}{col_format}"
            layout_parts.append(col_format)
    else:
        for key in column_keys:
            entry = headers[key]
            dark = entry.get("dark", False)
            if si_setup:
                col_format = f"S[{si_setup}]"
            else:
                col_format = "c"
            if dark:
                col_format = f">{{\\columncolor{{black!20}}}}{col_format}"
            layout_parts.append(col_format)
    layout = "|".join(layout_parts)
    layout = f"| {layout} |"

    # Print LaTeX table preamble
    print("\\begin{table}[h!]")
    print("    \\centering")
    if caption:
        print(f"    \\caption{{{caption}}}")
    if si_setup and not column_formats:
        print(f"    \\sisetup{{{si_setup}}}")  # Apply si_setup globally only if no column-specific format is provided
    print(f"    \\begin{{tabular}}{{{layout}}}")
    print("    \\toprule")

    # Print group headers if provided
    if header_groups:
        group_row = " & ".join(
            [f"\\multicolumn{{{span}}}{{c|}}{{{name}}}" for name, span in header_groups]
        )
        print(f"    {group_row} \\\\")
        print("    \\midrule")

    # Print main headers derived from headers values (using label)
    header_row = " & ".join(entry["label"] for entry in headers.values()) + " \\\\"
    print(f"    {header_row}")
    print("    \\midrule")

    # If data is a single dict, make it a list of one dict
    if isinstance(data, dict):
        data_blocks = [data]
    elif isinstance(data, list):
        data_blocks = data
    else:
        raise ValueError("Data must be a dictionary or a list of dictionaries.")

    # Process each data block
    for block_idx, data_block in enumerate(data_blocks):
        # Determine the maximum number of rows in this data block
        max_rows = 0
        for key in column_keys:
            entry = headers[key]
            # Check if data is provided in entry or in data block
            if 'data' in entry:
                column_data = entry['data']
            else:
                column_data = data_block.get(key, [])
            if isinstance(column_data, (list, np.ndarray)):
                length = len(column_data)
            else:
                length = 1  # Single value
            if length > max_rows:
                max_rows = length

        # Print data rows for this block
        for row in range(max_rows):
            row_data = []
            for key in column_keys:
                entry = headers[key]
                try:
                    # Get data for this column
                    if 'data' in entry:
                        column_data = entry['data']
                    else:
                        column_data = data_block.get(key, [])
                    # Determine the value
                    if isinstance(column_data, (list, np.ndarray)):
                        if row < len(column_data):
                            value = column_data[row]
                        else:
                            value = ""
                    else:
                        # Single value
                        repeat = entry.get('repeat', True)
                        if repeat:
                            value = column_data
                        else:
                            value = column_data if row == 0 else ""

                    # Check if value is a string
                    if isinstance(value, str):
                        formatted_value = f"{{{value}}}"  # Plain string with no error
                    elif value == "":
                        formatted_value = ""
                    else:
                        # Now handle errors
                        # First, check for error in data block
                        if 'err' in data_block and key in data_block['err']:
                            error = data_block['err'][key]
                            if isinstance(error, (list, np.ndarray)):
                                if row < len(error):
                                    error_value = error[row]
                                else:
                                    error_value = 0
                            else:
                                error_value = error
                        # Next, check for error in entry
                        elif "err" in entry:
                            error = entry["err"]
                            if isinstance(error, (list, np.ndarray)):
                                if row < len(error):
                                    error_value = error[row]
                                else:
                                    error_value = 0
                            else:
                                error_value = error
                        else:
                            error_value = 0

                        # Decide on intermed condition
                        intermed = entry.get("intermed", False)

                        # Use round_val to get rounded value and error
                        if error_value != 0:
                            rounded_val, err_round, power = round_val(value, err=error_value, intermed=intermed)

                            if power <= 0:
                                formatted_val_str = f"{rounded_val:.0f}"
                                formatted_err_str = f"{err_round:.0f}"
                            else:
                                formatted_val_str = f"{rounded_val:.{power}f}"
                                formatted_err_str = f"{err_round:.{power}f}"

                            formatted_value = "{$" + f"{formatted_val_str}" + " \\pm " + f"{formatted_err_str}" + "$}"
                        else:
                            rounded_val, power = round_val(value, err=0, intermed=intermed)

                            if power <= 0:
                                formatted_val_str = f"{rounded_val:.0f}"
                            else:
                                formatted_val_str = f"{rounded_val:.{power}f}"

                            formatted_value = "{$" + f"{formatted_val_str}" + "$}"

                except (IndexError, KeyError) as e:
                    formatted_value = "{Error}"
                    print(f"Error processing row {row}, column {key}: {e}")
                except Exception as e:
                    formatted_value = "{Error}"
                    print(f"Unexpected error processing row {row}, column {key}: {e}")

                row_data.append(formatted_value)

            # Create row string and print
            row_str = " & ".join(row_data) + " \\\\"
            print(f"    {row_str}")

        # If not the last data block, insert a midrule
        if block_idx < len(data_blocks) -1:
            print("    \\midrule")

    # Print table closing elements
    print("    \\bottomrule")
    print("    \\end{tabular}")
    if label:
        print(f"    \\label{{{label}}}")
    print("\\end{table}")

    return