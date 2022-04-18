import pandas as pd
import os
import glob
import yaml

def preprocess_data():
    # Get config variables
    config = read_yaml_config_file()
    use_dummy=bool(config['use_dummy'])
    filter_listings=bool(config['filter_listings'])
    year_start = int(config['year_start'])
    year_end = int(config['year_end'])
    years = [i for i in range(year_start, year_end+1)]
    # Set data path
    if use_dummy == True:
        listing_path = f'..\\..\\data\\dummy_data\\listings\\'
        loan_path = f'..\\..\\data\\dummy_data\\loans\\'
    else:
        listing_path = f'..\\..\\data\\real_data\\listings\\'
        loan_path = f'..\\..\\data\\real_data\\loans\\'
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
    return listing_df, loan_df


def read_files_to_pandas(filenames, cols = []):
    lis =[]
    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, encoding_errors='ignore')
        if len(cols)>0:
            cols_valid = []
            for col in cols:
                if col not in df.columns:
                    print(col)
                else:
                    cols_valid.append(col)
            df = df[cols_valid]
        lis.append(df)
    all_files_df = pd.concat(lis, axis=0, ignore_index=True)

    return all_files_df

def read_yaml_config_file():
    with open('column_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config