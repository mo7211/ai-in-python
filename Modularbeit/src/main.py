import logging
import pandas as pd
import os.path as osp
import time

from visualization import visualize
from utils import configurize_logger, log_versions
from data import clean_data, prep_data, SplitOption


def main():
    # log_versions()

    df = pd.read_csv(
        'Modularbeit/data/raw_data/Real Estate Dataset.csv', sep=';')

    configurize_logger(__name__)

    logging.info('Creating visualization before cleaning')
    visualize(df)

    logging.info('Clean data')
    split_option = SplitOption.WITH_INDEX
    cleaned_df = clean_data(df,split_option)

    logging.info('Creating visualization after cleaning')
    visualize(cleaned_df)

    preprocessed_df = prep_data(cleaned_df)

    visualize(preprocessed_df)


main()
