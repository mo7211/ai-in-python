# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_cleaning
from utils import configurize_logger
from data import clean_data, prep_data
from models import *
import config


def main():
    configurize_logger(__name__)
    show_plots = config.SHOW_PLOTS

    logging.info('Start script')

    df = pd.read_csv(
        config.INPUT_DATA_PATH, sep=";")

    visualize_cleaning(df, "before cleaning", show_plots)
    clean_data(df, config.SPLIT_OPTION)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)
    visualize_cleaning(cleaned_df, "after cleaning", show_plots)

    prep_data(cleaned_df)

    config.TARGET = 'price'

    preprocessed_df = pd.read_parquet(
        config.PREPROCESSED_DATA_PATH)

    visualize_cleaning(
        preprocessed_df, "after preprocessing", show_plots)

    # y = pd.read_csv(
    #     config.TARGET_DATA_PATH)
    # X = df = pd.read_hdf(config.FEATURE_DATA_PATH, 'df')
    # # pd.read_csv(config.FEATURE_DATA_PATH)

    # # drop columns 'name_msi', 'construction_type', 'district' from X
    # # X = X.drop(['name_nsi', 'construction_type', 'district'], axis='columns')

    # X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=config.TEST_SIZE, random_state=42)

    # sdg_regression(X_train, X_test, y_train, y_test)

    # regression(X_train, X_test, y_train, y_test)

    logging.info('Script succesfully ended')


if __name__ == "__main__":
    main()
