from pandas import DataFrame


def define_target(df:DataFrame, column_name:str) :
    y = df[column_name]

    # features
    X = df.drop(columns=[column_name])
    return y,X