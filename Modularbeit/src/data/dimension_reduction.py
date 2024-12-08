import logging
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA

from utils import plot_pca_explained_variance, plot_pca_best_projection
import config


def reduce_dimensions(df: DataFrame, reduce=True, n_components: float = 0.95):

    if reduce:
        logging.info('Reducing dimensions')
        n_columns_old = df.columns.size

        logging.info(f'n_components is {n_components}')
        pca = PCA(n_components=n_components)
        # pca = PCA()
        df_reduced = pca.fit_transform(df)

        # plot_pca_explained_variance(pca)
        # plot_pca_best_projection(df[['area','year_built']])

        logging.info(f'Reduced from {n_columns_old} to {
                     pca.n_components_} components')
        # logging.info(f'Features: {df_reduced.columns}')

        # Create a DataFrame from the reduced data
        df_reduced = pd.DataFrame(df_reduced,
                                  columns=[f'PC{i+1}' for i in range(df_reduced.shape[1])])

        logging.info(f'Reduced Features: {list(df_reduced.columns)}')

        return df_reduced
    return df
