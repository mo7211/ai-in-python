import pandas as pd

from visualization import visualize
from utils import Logger

logger  = Logger('Starting script')

logger.info('Importing raw data')
df = pd.read_csv('Modularbeit/data/raw_data/Real Estate Dataset.csv', sep=';')

logger.info('Creating visualization before cleaning')
visualize(df, logger)

logger.info('Cleaning raw data')

logger.info('Creating visualization after cleaning')
