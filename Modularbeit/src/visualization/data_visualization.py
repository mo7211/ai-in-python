# coding: utf-8

import logging

# Logger, create_scatterplot, calculate_null_ratios, create_barplot_null_values, create_barplot_year
from utils import *


def visualize_cleaning(df: DataFrame, title: str = '', show_plots:bool = True):
    if df is not None:
        title = ' ' + title if len(title) > 0 else ''
        # Visualization before cleaning
        logging.debug('Starting visualization' + title)

        # Visualize null ratios
        null_ratios = calculate_null_ratios(df)
        create_barplot_null_values(
            null_ratios, "Percentage of null values" + title, "Null values in %", "Columns", show_plots)

        # Visualize floors of assets
        create_scatterplot(
            df, 'floor', "Floor of asset" + title,  "# of floor", "Asset", [], show_plots)

        # Visualize year built
        create_barplot_year(df, 'year_built' , "Year of construction" + title,
                            "# of assets", "Year of construction", show_plots)

        # Visualize price
        create_scatterplot_price(
            df, 'price', "Real estate prices" + title, "Asset", "Price in Euro", show_plots)
    
