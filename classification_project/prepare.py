import pandas as pd
import numpy as np
# import splitting functions
from sklearn.model_selection import train_test_split
def prep_iris(df):
    """
    write meaningful docstrings
    """
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={"species_name":'species'})
    return df
def clean_titanic(df):
    """
    students - write docstring
    """
    #drop unncessary columns
    df = df.drop(columns=['embarked', 'age','deck', 'class'])
    #made this a string so its categorical
    df.pclass = df.pclass.astype(object)
    #filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')
    return df
def prep_telco(df):
    '''
    please write good docstrings
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    return df
def splitting_data(df, col):
    '''
    i will write a docstring, like send in target variable
    '''
    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=24,
                     stratify=df[col]
                    )
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=24,
                                      stratify=validate_test[col]
                                     )
    return train, validate, test

def preprocess_telco(telco_train, telco_val, telco_test):
    '''
    preprocess_telco will take in three pandas dataframes
    of our telco data, expected as cleaned versions of this 
    telco data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready versions of our clean data, with 
    columns sex and embark_town encoded in the one-hot fashion
    return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    '''
    # with a looping structure:
    # for df in [train_df, val_df, test_df]:
    #     df.drop(blah blah blah)
    #     df['pclass'] = df['pclass'].astype(int)
    
#     yesno_columns = ['paperless_billing', 'streaming_tv', 'streaming_movies', 'tech_support']
    
#     # Loop through each dataframe
#     for df in [telco_train, telco_val, telco_test]:
#         # Loop through each column that needs conversion
#         for col in yesno_columns:
#             # Replace 'Yes' with 1 and 'No' with 0 in the current column
#             df[col] = df[col].replace({'Yes': 1, 'No': 0})
    
#    telco_train = telco_train.drop(columns='customer_id')
    telco_train['monthly_charges'] = telco_train['monthly_charges'].astype(int)
#     telco_train['streaming_tv'] = telco_train['streaming_tv'].astype(int)
#     telco_train['streaming_movies'] = telco_train['streaming_movies'].astype(int)
#     telco_train['tech_support'] = telco_train['tech_support'].astype(int)
    
#    telco_val = telco_val.drop(columns='customer_id')
    telco_val['monthly_charges'] = telco_val['monthly_charges'].astype(int)
#     telco_val['streaming_tv'] = telco_val['streaming_tv'].astype(int)
#     telco_val['streaming_movies'] = telco_val['streaming_movies'].astype(int)
#     telco_val['tech_support'] = telco_val['tech_support'].astype(int)  
    
#    test_df = telco_test.drop(columns='customer_id')
    telco_test['monthly_charges'] = telco_test['monthly_charges'].astype(int)
#     telco_test['streaming_tv'] = telco_test['streaming_tv'].astype(int)
#     telco_test['streaming_movies'] = telco_test['streaming_movies'].astype(int)
#     telco_test['tech_support'] = telco_test['tech_support'].astype(int)
    
    
    
    encoding_var = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'internet_service_type', 'payment_type']
    
    encoded_dfs = []
    for df in [telco_train, telco_val, telco_test]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_var],
            drop_first=True
        ).astype(int)
        encoded_dfs.append(pd.concat(
            [df, df_encoded_cats],
            axis=1).drop(columns=encoding_var))
    
    return encoded_dfs
