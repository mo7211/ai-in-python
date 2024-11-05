

from pandas import DataFrame


def split_data(df:DataFrame):
    
        # # ## Split data-set


    # # Drop all where index values is null
    # df_with_index = df.dropna(axis=0, how='any', subset=[
    #                           'quality_of_living', 'safety', 'transport', 'services', 'relax', 'environment'], inplace=False)
    # minmax_scale = MinMaxScaler(feature_range=(0, 1))

    # scale_minmax(

    # columns = ['environment',
    #            'quality_of_living',
    #            'safety',
    #            'transport',
    #            'services',
    #            'relax']

    # def reshape_column(df_with_index, minmax_scale, column):
    #     df_with_index[column] = minmax_scale.fit_transform(
    #         df_with_index[column].values.reshape(-1, 1))
    #     return df_with_index

    # for column in columns:
    #     reshape_column(df_with_index, minmax_scale, column)
    #     print(column)

    # # df_with_index.to_csv('real_estate_index.csv')
    # # print(df_with_index)

    # # %%
    # # Drop index columns
    # df_without_index = df.drop(
    #     ['quality_of_living', 'safety', 'transport', 'services', 'relax', 'environment'], axis=1)



    # # df_without_index.to_csv('real_estate.csv')
    return df
