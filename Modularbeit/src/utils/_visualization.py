from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from utils._logging import get_time


def create_scatterplot(df, column_name, title, y_label, x_label, x_ticks, show_plot:bool=True):
    plt.scatter(list(df.index), df[column_name],
                color='blue', marker='x', alpha=0.1)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    save_fig(plt, title)
    if show_plot:
        plt.show()
    plt.clf()


def create_barplot_null_values(null_percentages, title, y_label, x_label, show_plot:bool=True):
    plt.bar(list(null_percentages.index), list(
        null_percentages.values), color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=45, ha='right')
    save_fig(plt, title)
    if show_plot:
        plt.show()
    plt.clf()


def create_barplot_null_values(null_percentages, title, y_label, x_label, show_plot:bool=True):

    plt.bar(list(null_percentages.index), list(
        null_percentages.values), color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=45, ha='right')
    save_fig(plt, title)
    if show_plot:
        plt.show()
    plt.clf()
    


def create_barplot_year(df: DataFrame, column: str, title: str, y_label: str, x_label: str, show_plot:bool=True) -> None:
    df_yb = df
    df_yb[column] = df_yb[column].replace(np.nan, -1, inplace=False)
    df_yb_grouped = df_yb.groupby(column)[column].count()

    plt.bar(list(df_yb_grouped.index[:-1]),
            list(df_yb_grouped.values[:-1]), 20, color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.yscale('log')
    save_fig(plt, title)
    if show_plot:
        plt.show()
    plt.clf()


def create_scatterplot_price(df: DataFrame, column: str, title: str, x_label: str, y_label: str, show_plot:bool=True):
    plt.scatter(list(df.index), df[column],
                color='blue', marker='o', alpha=0.1)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks([])
    save_fig(plt, title)
    if show_plot:
        plt.show()
    plt.clf()


def calculate_null_ratios(df: DataFrame):
    """
    Calculate the percentage of null values for each feature in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be analyzed.

    Returns:
        pd.Series: A Series containing the percentage of null values for each column.
    """
    df_is_null = df.isnull().sum().sort_values(ascending=False)

    return df_is_null / df.shape[0] * 100


IMAGES_PATH = Path('Modularbeit') / 'images' / \
    time.strftime("%Y-%m-%d_%H-%M-%S")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(plt, fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

    # # ## Visualize distribution of cellar, lift,

    # # cellar

    # plt.scatter( list(df.index), df['cellar'], color='blue',marker='x', alpha=0.1)
    # plt.title("Cellar")
    # plt.ylabel("Has cellar")
    # plt.xlabel("Asset")
    # plt.xticks([])
    # if show_plot:
        # plt.show()

    # # lift
    # plt.scatter( list(df.index), df['lift'], color='blue',marker='x', alpha=0.1)
    # plt.title("Real Estate Prices")
    # plt.ylabel("lift")
    # plt.xlabel("Asset")
    # plt.xticks([])
    # if show_plot:
        # plt.show()

    # # ## check Rooms

    # rooms = df.groupby('rooms')['rooms'].count()

    # plt.bar( list(rooms.index), list(rooms.values), .5)
    # plt.title("Real Estate Prices by # of rooms")
    # plt.ylabel("# of assets")
    # plt.xlabel("rooms")
    # plt.axis([0, 6, 0 , 7000])
    # plt.xticks([i for i in range(6)])
    # if show_plot:
        # plt.show()

    # #Mean price rooms
    # price_per_rooms = df.groupby('rooms')['price'].mean()
    # plt.scatter(list(price_per_rooms.index), list(price_per_rooms.values), marker='x')
    # plt.title("Real Estate Prices by # of rooms")
    # plt.ylabel("mean price")
    # plt.xlabel("rooms")
    # plt.xticks([i for i in range(6)])
    # if show_plot:
        # plt.show()

    # # ## Construction type
    # import numpy as np

    # construction_type_anmount = df.groupby('construction_type')['construction_type'].count().sort_values(ascending=False)

    # plt.bar( list(construction_type_anmount.index), list(construction_type_anmount.values), .5, color=plt.get_cmap('viridis')(np.linspace(0,1,construction_type_anmount.shape[0])))
    # plt.title("Assets by construction type")
    # plt.ylabel("# of assets")
    # plt.xlabel("construction type" )
    # plt.xticks(rotation=45)
    # if show_plot:
        # plt.show()

    # construction_type_price = df.groupby('construction_type')['price'].mean().sort_values(ascending=False)
    # plt.bar( list(construction_type_price.index), list(construction_type_price.values), .5, color=plt.get_cmap('viridis')(np.linspace(0,1,construction_type_anmount.shape[0])))
    # plt.title("Mean price by construction type")
    # plt.ylabel("price")
    # plt.xlabel("construction type" )
    # plt.xticks(rotation=45)
    # if show_plot:
        # plt.show()
