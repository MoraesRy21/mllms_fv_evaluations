import os
import pandas as pd
from matplotlib.figure import Figure

from utils.paths import PathBuilder


def save_plot(fig: Figure, plot_dir: PathBuilder, filename: str, plot_dpi=300, plot_bbox_inches="tight"):
    """
    Saves the given plot figure to the specified filename.

    This function saves a matplotlib figure to a file in the specified directory
    with a tight bounding box to minimize whitespace. After saving, it closes the
    figure to release memory.

    :param fig: The matplotlib figure to be saved.
    :type fig: matplotlib.figure.Figure
    :param plot_dir: PathBuilder to the directory where the plot will be saved.
    :type plot_dir: PathBuilder
    :param filename: The name of the file to save the figure to.
    :param plot_dpi: Resolution in dots per inch. Defaults to 300.
    :param plot_bbox_inches: How to handle bounding boxes. Defaults to "tight".
    :return: None
    :raises ValueError: If no filename is specified.
    """

    if filename is None:
        raise ValueError("Filename must be specified when saving a plot.")

    if not plot_dir.path.exists():
        os.makedirs(plot_dir.path)

    fig.savefig(plot_dir / filename, dpi=plot_dpi, bbox_inches=plot_bbox_inches)


def generate_csv(data_frame: pd.DataFrame, plot_dir: PathBuilder, filename: str, override: bool, **kwargs) -> None:
    """
    Generates a CSV file from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be exported
    file_path : str
        Full path of the CSV file to be created (including name and extension)
    **kwargs : optional arguments
        Additional parameters for pd.DataFrame.to_csv(), such as:
        - index: bool (default True) - whether to include the index
        - sep: str (default ',') - field separator
        - encoding: str (default 'utf-8') - file encoding
        - header: bool (default True) - whether to include header

    Usage examples:
    ---------------
    # Basic usage
    generate_csv(df, 'results.csv')

    # Without index
    generate_csv(df, 'results.csv', index=False)

    # With custom separator
    generate_csv(df, 'results.csv', index=False, sep=';')

    # With specific encoding
    generate_csv(df, 'results.csv', index=False, encoding='latin-1')
    """
    try:
        # Create directory if it doesn't exist
        if not plot_dir.path.exists():
            os.makedirs(plot_dir.path)

        if not override and (plot_dir.path / filename).exists():
            print(f"CSV file {filename} already exists. Skipping generation.")
            return
        # Set default values if not provided
        if 'index' not in kwargs:
            kwargs['index'] = False
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'

        # Export to CSV
        data_frame.to_csv((plot_dir.path / filename).__str__(), **kwargs)

        print(f"✅ CSV generated successfully: {filename}")
        print(f"📊 Total rows exported: {len(data_frame)}")
        print(f"📋 Total columns exported: {len(data_frame.columns)}")

    except Exception as e:
        print(f"❌ Error generating CSV: {str(e)}")
        raise


def extract_info_image_instance(filename) -> tuple:
    """
    Extracts the class identifier and photo index from the filename.
    Example filename:
    xx-xxxxx-xxxx-xxxxxxxxxx-xxxxxx-xxxxxxxxxxxxxx-xxxxxxxxxx-x.jpg
    OP-00027-20308-0000043019-005620-20251015131437-0022502834-3.jpg
    The second-to-last field is the class ID and the last one is the image index.
    """
    name, _ = os.path.splitext(filename)
    parts = name.split('-')
    if len(parts) != 8:
        return None

    operator_code: str = parts[0]
    code_x: str = parts[1]
    vehicle_number: str = parts[2]
    validator_number: str = parts[3]
    buss_line: str = parts[4]
    date_time: str = parts[5]
    internal_card_number: str = parts[6]  # class_id
    image_index: str = parts[7] # img_index

    return operator_code, code_x, vehicle_number, validator_number, buss_line, date_time, internal_card_number, image_index
