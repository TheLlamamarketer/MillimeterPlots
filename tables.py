from decimal import Decimal, ROUND_HALF_UP

def dec(value, max_precision=9, fixed_precision=None):
    decimal_value = Decimal(value)
    
    if fixed_precision is not None:
        precision = fixed_precision
    elif abs(decimal_value) < 1:
        precision = min(max_precision, abs(decimal_value.as_tuple().exponent) + 2)
    else:
        precision = max_precision - 2

    quantize_str = f"1.{'0' * precision}"
    rounded_value = decimal_value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    if fixed_precision is None:
        # Remove trailing zeros if no fixed precision is specified
        return str(rounded_value).rstrip('0').rstrip('.') if '.' in str(rounded_value) else str(rounded_value)
    else:
        # Keep trailing zeros to the specified fixed precision
        return f"{rounded_value:.{precision}f}"


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

    Parameters:
    - data (dict): Dictionary containing the data for the table.
    - headers (dict): Dictionary mapping data keys to a dictionary with "label" and "precision".
    - header_groups (list of tuples): Optional group headers in the form (group_name, number_of_columns).
    - caption (str): The caption for the table.
    - label (str): The label for referencing the table in LaTeX.
    - column_formats (list): List of specific `table-format` strings for each column.
    - si_setup (str): The siunitx package settings for numerical formatting for the entire table if no specific format is provided.
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
            [f"\\multicolumn{{{span}}}{{c|}}{{{name}}}" for name, span in header_groups]
        )
        print(f"    {group_row} \\\\")
        print("    \\midrule")

    # Print main headers derived from headers values (using label)
    header_row = " & ".join(entry["label"] for entry in headers.values()) + " \\\\"
    print(f"    {header_row}")
    print("    \\midrule")

    max_rows = max(len(data[key]) for key in headers)

    # Print data rows
    for row in range(max_rows):
        row_data = []
        for key, entry in headers.items():
            
            try:
                value = data[key][row] if row < len(data[key]) else ""
                
                fixed_precision = entry["precision"] if "precision" in entry else None
                if isinstance(fixed_precision, list):
                    fixed_precision = fixed_precision[row] if row < len(fixed_precision) else None

                # Format the value based on its type
                if isinstance(value, str):
                    formatted_value = "{" + value + "}"
                elif "err" in entry and isinstance(entry["err"], list):
                    error = entry["err"][row] if row < len(entry["err"]) else ""
                    formatted_value = "{$" + f"{dec(value, fixed_precision=fixed_precision)}" + " \\pm " + f"{dec(error, fixed_precision=fixed_precision)}" "$}"
                else:
                    formatted_value = "{$" + f"{dec(value, fixed_precision=fixed_precision)}" + "$}"


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

    # Print table closing elements
    print("    \\bottomrule")
    print("    \\end{tabular}")
    if label:
        print(f"    \\label{{{label}}}")
    print("\\end{table}")

# Example usage for a table with column-specific formatting:

# Example data dictionary
data_example = {
    "peak": ["CO 1", "CO 2", "CS 1", "Na 1", "Na 2", "Am 1", "Am 2"],
    "a": [3048.0, 3436.0, 1786.0, 1377.0, 3301.0, 80.2, 181.0],
    "b": [62.8, 63.1, 46.3, 42.1, 60.7, 28.2, 8.3],
    "E": [1.173, 1.333, 0.662, 0.511, 1.275, 0.027, 0.056],
}

# Mapping of LaTeX-formatted headers to data dictionary keys
header_to_key_map_example = {
    "{Peak}": "peak",
    "{$a$(\\#)}": "a",
    "{$b$(\\#)}": "b",
    "{E (MeV)}": "E",
}

# Column-specific formatting for table
column_formats_example = ["4.0", "4.2", "2.1", "0.2"]

# Caption and label for the table
caption_example = "Messdaten zur Analyse verschiedener Isotope"
label_example = "tab:IsotopAnalyse"

# Print a table with column-specific formatting
'''
print_standard_table(
    data=data_example,
    header_to_key_map=header_to_key_map_example,
    column_formats=column_formats_example,
    caption=caption_example,
    label=label_example,
)'''
