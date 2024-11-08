# coding: utf-8

import logging
import pandas as pd

from visualization import visualize_cleaning
from utils import configurize_logger
from data import clean_data, prep_data
import config


def main():
    configurize_logger(__name__)

    df = pd.read_csv(
        config.INPUT_DATA_PATH, sep=";")

    visualize_cleaning(df, "before cleaning", config.SHOW_PLOTS)
    clean_data(df, config.SPLIT_OPTION)

    cleaned_df = pd.read_csv(
        config.SPLITTED_DATA_PATH)
    # print(cleaned_df.head(4))
    visualize_cleaning(cleaned_df, "after cleaning", config.SHOW_PLOTS)

    preprocessed_df = prep_data(cleaned_df)

    visualize_cleaning(preprocessed_df, "after preprocessing")


if __name__ == "__main__":
    main()
