import pandas as pd
import os
import glob


def preprocess_data(data_path, year_start=2011, year_end=2012):

    listing_path = f'{data_path}/listings/'
    loan_path = f'{data_path}/loans/'
    listing_files = glob.glob(os.path.join(listing_path, "dummy*.csv"))
    loan_files = glob.glob(os.path.join(loan_path, "dummy*.csv"))
    print(listing_files, loan_files)
    lis, loan = [], []

    listing_df = read_files_to_pandas(listing_files)
    loan_df = read_files_to_pandas(loan_files)

    return listing_df, loan_df


def read_files_to_pandas(filenames):
    lis =[]
    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        lis.append(df)
    all_files_df = pd.concat(lis, axis=0, ignore_index=True)

    return all_files_df
