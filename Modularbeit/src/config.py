from pathlib import Path
import time
from utils._cleaning import SplitOption

CLEAN = True
PREPROCESS = True
TRAIN = False
TRAIN_METHOD = "xx"
TEST = False

SPLIT_OPTION = SplitOption.WITH_INDEX
SHOW_PLOTS = False

INPUT_DATA_PATH = 'Modularbeit/data/raw_data/Real Estate Dataset.csv'
CLEANED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned.csv'
SPLITTED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned_' + \
    SPLIT_OPTION.value + '.csv'


IMAGES_PATH = Path('Modularbeit') / 'images' / \
    time.strftime("%Y-%m-%d_%H-%M-%S")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
LOGGING_PATH = 'Modularbeit/logging'