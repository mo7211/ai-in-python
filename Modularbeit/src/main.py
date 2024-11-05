import logging 
import pandas as pd

from visualization import visualize
from utils import configurize_logger, log_versions
from data import clean_data

log_versions()

script = 'Starting script'

configurize_logger(script)

logging.info('Importing raw data')
df = pd.read_csv('Modularbeit/data/raw_data/Real Estate Dataset.csv', sep=';')

logging.info('Creating visualization before cleaning')
# visualize(df)

clean_data(df)

logging.info('Creating visualization after cleaning')
# visualize(df)
