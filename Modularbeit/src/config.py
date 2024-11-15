from pathlib import Path
import time
from utils._cleaning import SplitOption

# Options

CLEAN = False
PREPROCESS = False
TRAIN = True
TRAIN_METHOD = "xx"

SPLIT_OPTION = SplitOption.WITH_INDEX
SHOW_PLOTS = False
TARGET = ''
TEST_SIZE = 0.3

# Data

INPUT_DATA_PATH = 'Modularbeit/data/raw_data/Real Estate Dataset.csv'
CLEANED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned.csv'
SPLITTED_DATA_PATH = 'Modularbeit/data/cleaned_data/re_cleaned_' + \
    SPLIT_OPTION.value + '.csv'
PREPROCESSED_DATA_PATH = 'Modularbeit/data/features/re_preprosessed_' + \
                  SPLIT_OPTION.value + '.csv'
FEATURE_DATA_PATH = 'Modularbeit/data/features/re_features_' + TARGET + '_' +\
                  SPLIT_OPTION.value + '.csv'
TARGET_DATA_PATH = 'Modularbeit/data/features/re_target_' + TARGET + '_' +\
                  SPLIT_OPTION.value + '.csv'

# Paths

IMAGES_PATH = Path('Modularbeit') / 'images' / \
    time.strftime("%Y-%m-%d_%H-%M-%S")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
LOGGING_PATH = 'Modularbeit/logging'

# Preprossesing

FEATURE_MAPPER = {'condition': {'Development project': 1,
                                'Under construction': 2,
                                'Original condition': 3,
                                'Partial reconstruction': 4,
                                'Complete reconstruction': 5
                                },
                  'certificate': {'Unknown': 1,
                                  'none': 2,
                                  'G': 3,
                                  'F': 4,
                                  'E': 5,
                                  'D': 6,
                                  'C': 7,
                                  'B': 8,
                                  'A': 9
                                  }
                  }