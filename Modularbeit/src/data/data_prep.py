import logging

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

from utils import *


def prep_data(df: DataFrame):
    logging.info(50*"=")
    logging.info("Start data preprocessing")

    binarize_labels(df, "name_nsi")
    binarize_labels(df, "district")
    binarize_labels(df, 'construction_type')

    scale_minmax(df, 'price')
    df['area'] = df['area'].str.replace(',', '.').astype(float)
    scale_minmax(df, 'area')

    df['year_built'] = df['year_built'].astype(int)
    scale_minmax(df, 'year_built')

    df['last_reconstruction'] = df['last_reconstruction'].astype(int)
    scale_minmax(df, 'last_reconstruction')

    df['floor'] = df['floor'].astype(int)
    scale_minmax(df, 'floor')

    scale_minmax(df, 'rooms')

    condition_mapper = {'Development project': 1,
                        'Under construction': 2,
                        'Original condition': 3,
                        'Partial reconstruction': 4,
                        'Complete reconstruction': 5
                        }

    map_values(df, 'condition', condition_mapper)

    certificates_mapper = {'Unknown': 1,
                           'none': 2,
                           'G': 3,
                           'F': 4,
                           'E': 5,
                           'D': 6,
                           'C': 7,
                           'B': 8,
                           'A': 9
                           }

    map_values(df, 'certificate', condition_mapper)

    

    df.to_csv('Modularbeit/data/features/re_preprosessed.csv')


    return df
