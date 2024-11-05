import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def create_scatterplot(df, column_name, title, y_label, x_label, x_ticks):
    plt.scatter(list(df.index), df[column_name],
                color='blue', marker='x', alpha=0.1)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.show()

def create_barplot_null_values(null_percentages, title, y_label, x_label):
    plt.bar(list(null_percentages.index), list(
        null_percentages.values), color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def create_barplot_null_values(null_percentages, title, y_label, x_label):
    
    
    plt.bar(list(null_percentages.index), list(
        null_percentages.values), color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def create_barplot_year(df: DataFrame, column: str, title: str, y_label: str, x_label: str) -> None:
    df_yb = df
    df_yb[column] = df_yb[column].replace(np.nan, -1, inplace=False)
    df_yb_grouped = df_yb.groupby(column)[column].count()

    plt.bar(list(df_yb_grouped.index[:-1]),
            list(df_yb_grouped.values[:-1]), 20, color='blue')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.yscale('log')
    plt.show()

def create_scatterplot_price(df:DataFrame, column:str, title:str, x_label:str, y_label:str):
    plt.scatter( list(df.index), df[column], color='blue',marker='o', alpha=0.1)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks([])
    plt.show()


def calculate_null_ratios(df:DataFrame):
    """
    Calculate the percentage of null values for each feature in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be analyzed.

    Returns:
        pd.Series: A Series containing the percentage of null values for each column.
    """
    df_is_null = df.isnull().sum().sort_values(ascending=False)
    return df_is_null / df.shape[0] * 100
