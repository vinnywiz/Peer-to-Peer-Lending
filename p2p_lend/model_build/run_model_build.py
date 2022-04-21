import os
import pandas as pd
import yaml
import numpy as np
import pickle

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

MODEL_TYPES = [
    'Logistic',
    'RandomForest',
    'XGBoost'
]

def run_model_build(model_type="Logistic", data_path=None, apply_smote=None, use_dummy=None, processed_file='loan_listing_cleaned.csv'):

    assert(model_type in MODEL_TYPES)

    sk_model = None
    # Read config
    config = read_yaml_config_file()
    # Define variables from config
    model_X_columns = config['model_X_columns']
    model_y_column = config['model_y_column']
    data_path = config['data_path']
    if use_dummy is None:
        use_dummy = config['use_dummy']
    if apply_smote is None:
        apply_smote = config['apply_smote']
    # Load Processed Data
    processed_df = read_processed_data(data_path=data_path, use_dummy=use_dummy, file_name=processed_file)
    #print(model_columns)
    processed_df = processed_df[model_X_columns]
    # Factorize columns
    processed_df = factorize_loan_listing_data(processed_df)
    write_data_frame_for_analysis(processed_df, use_dummy=use_dummy, data_path=data_path)
    # Train Test Split data
    X = processed_df.drop(model_y_column, axis=1).copy()
    y = processed_df[model_y_column].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state = 42)
    # feature scaling
    scaler = RobustScaler(copy=True) # robust scaler takes care of outliers better  
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    # Apply SMOTE sampling
    if apply_smote and (not (use_dummy)):
        print('applied_smote')
        X_train, y_train = resample_using_SMOTE(X_train, y_train)
    # Build Model
    print(X_train.shape, y_train.shape)
    sk_model = run_model(X_train, y_train,X_test, y_test, model_type=model_type)
    # Save Model
    save_trained_model(sk_model=sk_model, X_test=X_test, y_test=y_test, data_path=data_path, use_dummy=use_dummy)
    return sk_model

def resample_using_SMOTE(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.7)  

    # chain transforms into a pipeline
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # fit and apply pipeline to dataset
    X_smt, y_smt = pipeline.fit_resample(X_train, y_train)
    return X_smt, y_smt

def run_model(X_train, y_train,X_test, y_test, model_type=None):
    if model_type=='Logistic':
        sk_model = build_logistic_regression(X_train,y_train)
    elif model_type=='RandomForest':
        sk_model = build_random_forest(X_train,y_train)
    elif model_type=='XGBoost':
        sk_model = build_xg_boost(X_train,y_train,X_test, y_test)
    else:
        sk_model = build_logistic_regression(X_train,y_train)
    return sk_model

def save_trained_model(sk_model,X_test, y_test, data_path=None, use_dummy=None):
    if use_dummy:
        file_path = os.path.join(data_path, 'dummy_data','model')
    else:
        file_path = os.path.join(data_path, 'real_data','model')
    with open(os.path.join(file_path,'trained_model.pkl'), 'wb') as f:
        pickle.dump(sk_model, f)
    pd.DataFrame(X_test).to_csv(os.path.join(file_path, 'X_test.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(file_path, 'y_test.csv'), index=False)
    return None

def factorize_loan_listing_data(X):
    # factorize fico range and rename column
    X['TUFicoRange'] = pd.factorize(X['TUFicoRange'])[0]
    X.rename(columns={'TUFicoRange':'FicoRange'}, inplace=True)

    # take fico out of the data so it's not dummy and re add after that
    fico = X['FicoRange']
    X.drop('FicoRange', axis=1, inplace=True)

    # make dummy variables for categorical columns for logistic regression
    X = pd.get_dummies(X)

    # scale co_borrower to 1 and 0
    X['co_borrower_application'] = np.where(X['co_borrower_application']==True, 1, 0)

    # re add fico
    X['FicoRange'] = fico

    return X

def read_processed_data(data_path='data', use_dummy=True, file_name='loan_listing_cleaned.csv'):
    df=None
    if use_dummy:
        file_path = os.path.join(data_path, 'dummy_data','processed_data',file_name)
    else:
        file_path = os.path.join(data_path, 'real_data','processed_data',file_name)
    df = pd.read_csv(file_path, index_col=None, header=0, low_memory=False, encoding_errors='ignore')
    return df

def build_logistic_regression(X_train,y_train):
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(max_iter = 300, 
                                n_jobs = -1, random_state=42)
    lr_model.fit(X_train, y_train)
    print(lr_model.score(X_train, y_train))
    return lr_model

def build_random_forest(X_train,y_train):
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=42)
    rf_model.fit(X_train,y_train)

    print(rf_model.score(X_train, y_train))
    return None

def build_xg_boost(X_train,y_train,X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report
    xgb_model = XGBClassifier(n_estimators=50, max_depth=4) 
    xgb_model.fit(X_train,y_train) 

    print(xgb_model.score(X_train, y_train))
    xgb_prdict = xgb_model.predict(X_test) 
    print(classification_report(y_test, xgb_prdict))
    return None

def read_yaml_config_file():
    yaml_path = os.path.join('p2p_lend','model_build', 'model_config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def write_data_frame_for_analysis(df, use_dummy=True, data_path=None):
    if use_dummy:
        file_path = os.path.join(data_path, 'dummy_data','analysis','analysis_df.csv')
    else:
        file_path = os.path.join(data_path, 'real_data','analysis','analysis_df.csv')
    df.to_csv(file_path)
    return None