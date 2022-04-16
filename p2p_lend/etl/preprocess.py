import pandas as pd
import os


def preprocess_data(data_path, year_start=2011, year_end=2012):

    listing_df = pd.read_csv(f'{data_path}/listings/', encoding_errors='ignore', low_memory=False)
    loan_df = pd.read_csv(f'{data_path}/loans/', encoding_errors='ignore', low_memory=False)