# coding: utf-8

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logger, create_scatterplot, calculate_null_ratios, create_barplot_null_values, create_barplot_year
from utils import *


def visualize(df):
    # Visualization before cleaning
    logging.debug('Starting visualization')

    # Visualize null ratios

    null_ratios = calculate_null_ratios(df)
    create_barplot_null_values(
        null_ratios, "Percentage of null values", "Null values in %", "Columns")

    # Visualize floors of assets
    create_scatterplot(
        df, 'floor', "Floor of asset before cleaning",  "# of floor", "Asset", [])

    # Visualize year built
    create_barplot_year(df, 'year_built', "Year of construction",
                        "# of assets", "Year of construction")

    # Visualize price
    create_scatterplot_price(
        df, 'price', "Real estate prices before cleaning", "Asset", "Price in Euro")

    # # ## Visualize distribution of cellar, lift,

    # # cellar

    # plt.scatter( list(df.index), df['cellar'], color='blue',marker='x', alpha=0.1)
    # plt.title("Cellar")
    # plt.ylabel("Has cellar")
    # plt.xlabel("Asset")
    # plt.xticks([])
    # plt.show()

    # # lift
    # plt.scatter( list(df.index), df['lift'], color='blue',marker='x', alpha=0.1)
    # plt.title("Real Estate Prices")
    # plt.ylabel("lift")
    # plt.xlabel("Asset")
    # plt.xticks([])
    # plt.show()

    # # ## check Rooms

    # rooms = df.groupby('rooms')['rooms'].count()

    # plt.bar( list(rooms.index), list(rooms.values), .5)
    # plt.title("Real Estate Prices by # of rooms")
    # plt.ylabel("# of assets")
    # plt.xlabel("rooms")
    # plt.axis([0, 6, 0 , 7000])
    # plt.xticks([i for i in range(6)])
    # plt.show()

    # #Mean price rooms
    # price_per_rooms = df.groupby('rooms')['price'].mean()
    # plt.scatter(list(price_per_rooms.index), list(price_per_rooms.values), marker='x')
    # plt.title("Real Estate Prices by # of rooms")
    # plt.ylabel("mean price")
    # plt.xlabel("rooms")
    # plt.xticks([i for i in range(6)])
    # plt.show()

    # # ## Construction type
    # import numpy as np

    # construction_type_anmount = df.groupby('construction_type')['construction_type'].count().sort_values(ascending=False)

    # plt.bar( list(construction_type_anmount.index), list(construction_type_anmount.values), .5, color=plt.get_cmap('viridis')(np.linspace(0,1,construction_type_anmount.shape[0])))
    # plt.title("Assets by construction type")
    # plt.ylabel("# of assets")
    # plt.xlabel("construction type" )
    # plt.xticks(rotation=45)
    # plt.show()

    # construction_type_price = df.groupby('construction_type')['price'].mean().sort_values(ascending=False)
    # plt.bar( list(construction_type_price.index), list(construction_type_price.values), .5, color=plt.get_cmap('viridis')(np.linspace(0,1,construction_type_anmount.shape[0])))
    # plt.title("Mean price by construction type")
    # plt.ylabel("price")
    # plt.xlabel("construction type" )
    # plt.xticks(rotation=45)
    # plt.show()
