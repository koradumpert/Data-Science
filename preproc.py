import numpy as np
import pandas as pd

def load_data(file_path_prefix="/Users/koradumpert/Desktop/My_Stuff/DSCI372/project_4_ML/"):
    """
    Load the data for processing
    :param filename: The location of the data to load from file.
    :return: cleaned DF of spam/not spam emails
    """


    df = pd.read_csv(file_path_prefix + "WELFake_Dataset.csv",header = 0, index_col = 0)
    df = df.dropna().reset_index(drop = True)

    '''
    mapper = DataFrameMapper([(df.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(df.copy(), 13)
    scaled_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    '''
    df.to_csv(file_path_prefix + 'cleaned_FNews_data.csv', encoding='utf-8', index = False)





if __name__ == "__main__":
    # Run the experiment
    load_data()