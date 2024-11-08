

from pandas import DataFrame
from sklearn.naive_bayes import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler


def binarize_labels(df: DataFrame, column: str):
    names = df[column]
    feature_name_nsi = names.values.reshape(-1, 1)

    one_hot = LabelBinarizer()
    df[column] = list(one_hot.fit_transform(feature_name_nsi))


def scale_minmax(df: DataFrame, column: str):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    df[column] = minmax_scale.fit_transform(df[column].values.reshape(-1, 1))
    return minmax_scale


def map_values(df: DataFrame, column: str, condition_mapper: dict):
    df[column] = df[column].replace(condition_mapper)

    # # ## Split data-set
    # minmax_scale = MinMaxScaler(feature_range=(0, 1))

    # columns = ['environment',
    #         'quality_of_living',
    #         'safety',
    #         'transport',	
    #         'services',	
    #         'relax']

    # def reshape_column(df_with_index, minmax_scale, column):
    #     df_with_index[column] = minmax_scale.fit_transform(df_with_index[column].values.reshape(-1, 1))
    #     return df_with_index

    # for column in columns:
    #     reshape_column(df_with_index, minmax_scale, column)
    #     print(column)
