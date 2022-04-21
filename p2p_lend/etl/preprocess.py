import pandas as pd
import numpy as np
import os
import glob
import yaml

def preprocess_data(use_dummy=None, data_path=None):
    # Get config variables
    config = read_yaml_config_file()
    if use_dummy is None:
        use_dummy=bool(config['use_dummy'])
    filter_listings=bool(config['filter_listings'])
    year_start = int(config['year_start'])
    year_end = int(config['year_end'])
    years = [i for i in range(year_start, year_end+1)]
    if data_path is None:
        data_path = config['data_path']
    # Set data path
    if use_dummy == True:
        listing_path = os.path.join(data_path, 'dummy_data','listings')
        loan_path = os.path.join(data_path, 'dummy_data','loans')
    else:
        listing_path = os.path.join(data_path, 'real_data','listings')
        loan_path = os.path.join(data_path, 'real_data','loans')
    listing_files = glob.glob(os.path.join(listing_path, "*.*"))
    loan_files = glob.glob(os.path.join(loan_path, "*.*"))
    if not use_dummy:
        listing_files = [file for file in listing_files if int(file[:-4][-4:]) in years]
        loan_files = [file for file in loan_files if int(file[:-4][-4:]) in years]
    #print(listing_files, loan_files)
    if filter_listings:
        config = read_yaml_config_file()
        listing_columns = config['listing_columns']
        listing_df = read_files_to_pandas(listing_files, listing_columns)
    else:
        listing_df = read_files_to_pandas(listing_files)
    loan_df = read_files_to_pandas(loan_files)
    # Join loan & listings
    loan_listing_df = join_listings_and_loans(loan_df, listing_df)
    # Data type conversion Imputing variables
    cleaned_df = clean_loan_listings_data(loan_listing_df)
    # write to file
    write_file(use_dummy, cleaned_df, data_path, file_name=f'loan_listing_cleaned.csv')
    return listing_df, loan_df, cleaned_df#, cleaned_df

def write_file(use_dummy, df, data_path, file_name=f'loan_listing_cleaned.csv'):
    if use_dummy:
        file_path = os.path.join(data_path, 'dummy_data','processed_data', file_name)
        df.to_csv(file_path, index=False)
    else:
        file_path = os.path.join(data_path, 'real_data','processed_data', file_name)
        df.to_csv(file_path, index=False)
    return 'Success'

def join_listings_and_loans(loan_df, listing_df):
    '''`
    Merge the two datasets
    '''
    loan_listing_df = pd.merge(loan_df, listing_df,  
        how='left', 
        left_on=['origination_date','amount_borrowed','borrower_rate','prosper_rating','term','co_borrower_application'],         
        right_on = ['loan_origination_date','amount_funded','borrower_rate','prosper_rating','listing_term','CoBorrowerApplication'])
    # Find Valid Loans
    loan_listing_valid_df = loan_listing_df.groupby('loan_number')['origination_date'].count().reset_index()
    loan_listing_valid_df['validity'] = np.where(loan_listing_valid_df['origination_date']>1, 'Invalid', 'Valid')
    loan_listing_valid_df = loan_listing_valid_df[['loan_number','validity']]
    # Filter to valid loans
    loan_listing_valid_df = pd.merge(loan_listing_df,loan_listing_valid_df, on='loan_number', how='inner')
    loan_listing_valid_df = loan_listing_valid_df[loan_listing_valid_df['validity']=='Valid']
    return loan_listing_valid_df

def clean_loan_listings_data(df):
    # define bad loan
    df.loc[((df['loan_status_description']=='CHARGEOFF')),'bad_loan'] = 1
    df.loc[((df['loan_status_description']=='DEFAULTED')),'bad_loan'] = 1
    df.loc[((df['loan_status_description']=='CANCELLED')),'bad_loan'] = 1
    df.loc[((df['loan_status_description']=='COMPLETED')),'bad_loan'] = 0
    df.loc[((df['loan_status_description']=='CURRENT')),'bad_loan'] = 0
    # convert date columns to datetime
    df['listing_start_date'] = pd.to_datetime(df['listing_start_date'], format="%Y-%m-%d %H:%M:%S")
    df['listing_end_date'] = pd.to_datetime(df['listing_end_date'], format="%Y-%m-%d %H:%M:%S")
    df['TUFicoDate'] = pd.to_datetime(df['TUFicoDate'], format="%Y-%m-%d %H:%M:%S")
    # stated_monthly_income
    df['stated_monthly_income'].fillna(df['stated_monthly_income'].median(), inplace=True)
    # listing monthly payment
    df['listing_monthly_payment'].fillna(df['listing_monthly_payment'].median(), inplace=True)
    # occupation
    df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
    # months employed
    df['months_employed'].fillna(df['months_employed'].median(), inplace=True)
    # prosper score
    df['prosper_score'].fillna(df['prosper_score'].mode()[0], inplace=True)
    # select rows where date is not nan
    df = df[df['listing_start_date'].notna()]
    df = df[df['listing_end_date'].notna()]
    # borrower state
    df['borrower_state'].fillna(df['borrower_state'].mode()[0], inplace=True)
    df['loan_status'] = np.where(df['loan_status_description']=='COMPLETED', 1, 0)
    # renaming some columns
    df.rename(columns={'listing_monthly_payment': 'monthly_payment', 'stated_monthly_income': 'monthly_income'}, inplace=True)
    # Feature Engineering
    df['EMI'] = df['amount_borrowed'] / df['term']
    df['balance_income'] = df['monthly_income'] - df['EMI']
    #df['balance_income_log'] = np.log(df['balance_income'])
    df.drop('loan_status_description', axis=1, inplace=True) # drop duplicated column
    return df

def read_files_to_pandas(filenames, cols = []):
    lis =[]
    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, encoding_errors='ignore')
        if len(cols)>0:
            cols_valid = []
            for col in cols:
                if col not in df.columns:
                    continue
                    #print(col)
                else:
                    cols_valid.append(col)
            df = df[cols_valid]
        lis.append(df)
    all_files_df = pd.concat(lis, axis=0, ignore_index=True)

    return all_files_df

def read_yaml_config_file():
    yaml_path = os.path.join('p2p_lend','etl', 'column_config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config