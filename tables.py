def print_standard_table(
    data,
    headers,
    header_to_key_map,
    header_groups=None,
    caption=None,
    label=None,
    precision=None,
    column_formats=None,
    si_setup=None,
):
    """
    Prints a standard LaTeX table with optional group headers, specified precision, and column-specific formats.

    Parameters:
    - data (dict): Dictionary containing the data for the table. Each key corresponds to a column name.
    - headers (list): List of LaTeX formatted headers (e.g., "{$h_0$ (mm)}").
    - header_to_key_map (dict): Dictionary mapping headers to corresponding keys in the data.
    - header_groups (list of tuples): Optional group headers in the form (group_name, number_of_columns). For example: [("Group 1", 3), ("Group 2", 2)].
    - caption (str): The caption for the table.
    - label (str): The label for referencing the table in LaTeX.
    - precision (dict): Dictionary specifying the precision for each column (e.g., {"column1": 2, "column2": 3}).
    - column_formats (list): List of specific `table-format` strings for each column, e.g., ["4.0", "4.2", "2.2", "1.2"].
    - si_setup (str): The siunitx package settings for numerical formatting for the entire table if no specific format is provided for columns.
    """
    # Number of columns
    num_columns = len(headers)

    # Create layout for tabular environment based on column-specific formats if provided
    if column_formats:
        layout = "|".join([f"S[table-format={fmt}]" for fmt in column_formats])
    else:
        # Use default si_setup for all columns if column-specific formats are not provided
        layout = (
            "|".join([f"S[{si_setup}]" for _ in range(num_columns)])
            if si_setup
            else "|".join(["c" for _ in range(num_columns)])
        )

    layout = f"| {layout} |"

    # Print LaTeX table preamble
    print("\\begin{table}[h!]")
    print("    \\centering")
    if caption:
        print(f"    \\caption{{{caption}}}")
    if si_setup and not column_formats:
        print(
            f"    \\sisetup{{{si_setup}}}"
        )  # Apply si_setup globally only if no column-specific format is provided
    print(f"    \\begin{{tabular}}{{{layout}}}")
    print("    \\toprule")

    # Print group headers if provided
    if header_groups:
        group_row = " & ".join(
            [f"\\multicolumn{{{span}}}{{c|}}{{{name}}}" for name, span in header_groups]
        )
        print(f"    {group_row} \\\\")
        print("    \\midrule")

    # Print main headers
    header_row = " & ".join(headers) + " \\\\"
    print(f"    {header_row}")
    print("    \\midrule")

    # Determine number of rows in the table based on the length of the first column in data
    first_key = list(data.keys())[0]
    num_rows = len(data[first_key])

    # Print data rows
    for row in range(num_rows):
        row_data = []
        for header in headers:
            # Retrieve the corresponding key for this header
            col_key = header_to_key_map[header]

            # Retrieve the value for this row and column using the key
            value = data[col_key][row]

            # Format the value based on the precision, if specified
            if precision and col_key in precision:
                formatted_value = f"{value:.{precision[col_key]}f}"
            else:
                formatted_value = f"{value}"

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

# LaTeX formatted headers
headers_example = [
    "{Peak}",
    "{$a$(\\#)}",
    "{$b$(\\#)}",
    "{E (MeV)}",
]

# Mapping of headers to data dictionary keys
header_to_key_map_example = {
    "{Peak}": "peak",
    "{$a$(\\#)}": "a",
    "{$b$(\\#)}": "b",
    "{E (MeV)}": "E",
}

# Column-specific formatting for table
column_formats_example = ["4.0", "4.2", "2.2", "1.2"]

# Caption and label for the table
caption_example = "Messdaten zur Analyse verschiedener Isotope"
label_example = "tab:IsotopAnalyse"

# Print a table with column-specific formatting
print_standard_table(
    data=data_example,
    headers=headers_example,
    header_to_key_map=header_to_key_map_example,
    column_formats=column_formats_example,
    caption=caption_example,
    label=label_example,
)


# Still a very work in progress