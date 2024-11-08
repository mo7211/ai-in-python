# coding: utf-8

import logging

# Logger, create_scatterplot, calculate_null_ratios, create_barplot_null_values, create_barplot_year
from utils import *


def visualize(df: DataFrame):
    # Visualization before cleaning
    logging.debug('Starting visualization')

    # Visualize null ratios

    null_ratios = calculate_null_ratios(df)

    # fix case no null ratios
    create_barplot_null_values(
        null_ratios, "Percentage of null values", "Null values in %", "Columns")

    # Visualize floors of assets
    create_scatterplot(
        df, 'floor', "Floor of asset",  "# of floor", "Asset", [])

    # Visualize year built
    create_barplot_year(df, 'year_built', "Year of construction",
                        "# of assets", "Year of construction")

    # Visualize price
    create_scatterplot_price(
        df, 'price', "Real estate prices", "Asset", "Price in Euro")
