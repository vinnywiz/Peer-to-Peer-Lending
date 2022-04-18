import pandas as pd
import os
import glob
import yaml

def preprocess_data(year_start=2011, year_end=2012, filter_listings=True, use_dummy=True):
    if use_dummy == True:
        listing_path = f'..\\..\\data\\dummy_data\\listings\\'
        loan_path = f'..\\..\\data\\dummy_data\\loans\\'
    else:
        listing_path = f'..\\..\\data\\real_data\\listings\\'
        loan_path = f'..\\..\\data\\real_data\\loans\\'        
    listing_files = glob.glob(os.path.join(listing_path, "dummy*.csv"))
    loan_files = glob.glob(os.path.join(loan_path, "dummy*.csv"))
    print(listing_files, loan_files)
    listing_df = read_files_to_pandas(listing_files)
    loan_df = read_files_to_pandas(loan_files)
    if filter_listings:
        config = read_yaml_config_file()
        listing_columns = config['listing_columns']
        listing_columns_valid = []
        for col in listing_columns:
            if col not in listing_df.columns:
                print(col)
            else:
                listing_columns_valid.append(col)
        listing_df = listing_df[listing_columns_valid]
    return listing_df, loan_df


def read_files_to_pandas(filenames):
    lis =[]
    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        lis.append(df)
    all_files_df = pd.concat(lis, axis=0, ignore_index=True)

    return all_files_df

def read_yaml_config_file():
    with open('column_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config