import logging 
import pandas as pd

from visualization import visualize
from utils import configurize_logger, log_versions

log_versions()

script = 'Starting script'

with script:
    configurize_logger(script)

    logging.info('Importing raw data')
    df = pd.read_csv('Modularbeit/data/raw_data/Real Estate Dataset.csv', sep=';')

    logging.info('Creating visualization before cleaning')
    visualize(df)

    logging.info('Cleaning raw data')

    logging.info('Creating visualization after cleaning')
