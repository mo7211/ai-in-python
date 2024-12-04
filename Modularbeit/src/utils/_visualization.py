from pathlib import Path
import time
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import pydotplus
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor

import config
from utils._logging import get_time


def create_scatterplot(df, column_name, title, y_label, x_label, x_ticks, show_plot: bool = True):
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


def create_barplot_null_values(null_percentages, title, y_label, x_label, show_plot: bool = True):
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


def create_barplot_null_values(null_percentages, title, y_label, x_label, show_plot: bool = True):

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


def create_barplot_year(df: DataFrame, column: str, title: str, y_label: str, x_label: str, show_plot: bool = True) -> None:
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


def create_scatterplot_price(df: DataFrame, column: str, title: str, x_label: str, y_label: str, show_plot: bool = True):
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


def save_fig(plt, fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = config.IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def visualize_model():
    return True


def visualize_sdg_regressor(y: DataFrame, X: DataFrame, model: SGDRegressor, column_name: str):
    X_area = X[[column_name]].values  # Convert to numpy array
    y_price = y.values

    # Calculate the mean of each feature
    means = X.mean().values

    # Create a new dataset where each column has its mean, except for the column we vary
    X_modified = np.tile(means, (X_area.shape[0], 1))
    X_modified[:, X.columns.get_loc(column_name)] = X_area.flatten()

    # # Prepare a new dataset where 'area' varies and other features are set to their mean
    # X_modified = np.zeros((X_area.shape[0], X.shape[1]))
    # X_modified[:, X.columns.get_loc(column_name)] = X_area.flatten()

    # Predictions based on the new modified dataset
    y_predicted = model.predict(X_modified)

    # Scatterplot of the actual data
    plt.scatter(X_area, y_price, color='blue', label='Actual Prices')

    # Plot the prediction line
    order = np.argsort(X_area.flatten())  # For a sorted prediction line
    plt.plot(X_area[order], y_predicted[order],
             color='red', label='Prediction')

    plt.xlabel(column_name)
    plt.ylabel('Price')
    title = f'SGDRegressor Prediction Visualization on "{column_name}"'
    plt.title(title)
    plt.legend()
    save_fig(plt, title)
    plt.show()


def create_pairplot(df: DataFrame):

    g = pd.plotting.scatter_matrix(df, figsize=(10, 10), marker='.', hist_kwds={
                                   'bins': 10}, s=60, alpha=0.8, range_padding=0.1)
    for ax in g[:, 0]:  # Iterate over the first column of subplot axes
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.show()


def create_heatmap(df: DataFrame):
    # Create a heatmap
    plt.imshow(df, cmap='hot', interpolation='nearest')

    # Add a color bar
    plt.colorbar()

    # Show the plot
    plt.show()


def plot_tree(pipeline, X):
    decision_tree = pipeline.named_steps['tree']

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(decision_tree, feature_names=X.columns, filled=True)
    plt.show()


def plot_pca_explained_variance(pca):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1  # d equals 154#

    plt.figure(figsize=(6, 4))
    plt.plot(cumsum, linewidth=3)
    plt.axis([0, 170, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, 0.95], "k:")
    plt.plot([0, d], [0.95, 0.95], "k:")
    plt.plot(d, 0.95, "ko")
    plt.annotate("Elbow", xy=(26, 0.95), xytext=(70, 0.7),
                 arrowprops=dict(arrowstyle="->"))
    plt.grid(True)
    title = "explained_variance_plot"
    save_fig(plt, title)
    plt.show()
    # # ## Visualize distribution of cellar, lift,


def plot_pca_best_projection(df: DataFrame):

    np.random.seed(3)
    X_line = df.values
    m = X_line.shape[0]

    angle = np.pi / 5

    u1 = np.array([np.cos(angle), np.sin(angle)])
    u2 = np.array([np.cos(angle - 2 * np.pi / 6),
                  np.sin(angle - 2 * np.pi / 6)])
    u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])

    X_proj1 = X_line @ u1.reshape(-1, 1)
    X_proj2 = X_line @ u2.reshape(-1, 1)
    X_proj3 = X_line @ u3.reshape(-1, 1)

    plt.figure(figsize=(8, 4))
    plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.plot([-1.4, 1.4], [-1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], "k-",
             linewidth=2)
    plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], "k--",
             linewidth=2)
    plt.plot([-1.4, 1.4], [-1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], "k:",
             linewidth=2)
    plt.plot(X_line[:, 0], X_line[:, 1], "ro", alpha=0.5)
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=4, alpha=0.9,
              length_includes_head=True, head_length=0.1, fc="b", ec="b", zorder=10)
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=1, alpha=0.9,
              length_includes_head=True, head_length=0.1, fc="b", ec="b", zorder=10)
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", color="blue")
    plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", color="blue")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)
    # plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.grid()

    plt.subplot2grid((3, 2), (0, 1))
    plt.plot([-2, 2], [0, 0], "k-", linewidth=2)
    plt.plot(X_proj1[:, 0], np.zeros(m), "ro", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    # plt.axis([-2, 2, -1, 1])
    plt.grid()

    plt.subplot2grid((3, 2), (1, 1))
    plt.plot([-2, 2], [0, 0], "k--", linewidth=2)
    plt.plot(X_proj2[:, 0], np.zeros(m), "ro", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    # plt.axis([-2, 2, -1, 1])
    plt.grid()

    plt.subplot2grid((3, 2), (2, 1))
    plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), "ro", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    # plt.axis([-2, 2, -1, 1])
    plt.xlabel("$z_1$")
    plt.grid()

    save_fig(plt, "pca_best_projection_plot")
    plt.show()


def plot_feature_importance(model: BaseEstimator, X: DataFrame):
    # feature importance calculation
    importances = model.feature_importances_

    # Sort feature importance with decreasing values
    indices = np.argsort(importances)[::-1]

    # Sort feature names according to feature importance
    names = [X.columns[i] for i in indices]

    # generate the diagram
    plt.figure()

    # generate the diagram title
    plt.title("Feature importance")

    # add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # feature name as name on the x-axis
    plt.xticks(range(X.shape[1]), names, rotation=90)

    # show diagram
    plt.show()

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
