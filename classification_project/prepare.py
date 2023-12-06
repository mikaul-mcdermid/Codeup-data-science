import pandas as pd
import numpy as np
# import splitting functions
from sklearn.model_selection import train_test_split
def prep_iris(df):
    """
    Prep iris function takes in the dataframe and adjusts it via dropping columns "species_id" and "measurement_id" and renaming the column "species_name" to "species". Once cleaned, the dataframe is returned.
    """
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={"species_name":'species'})
    return df
def clean_titanic(df):
    """
    Clean titanic function takes in the dataframe and adjusts it via dropping columns "embarked", "age", "deck" and "class". It changes the column "pclass" to an object dtype and finally adjusts the "embark_town" column by filling invalid entry types with the mode of the column. Once cleaned, the dataframe is returned and can be renamed for further analysis.
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
    Prep telco function takes in the dataframe and adjusts it via dropping columns "payment_type_id", "internet_service_type_id", and "contract_type_id". It accounts for an issue in the "total_charges" column by adjusting a space value and converting it to a float so that the dtype can be adjusted. Once cleaned, the dataframe is returned and can be renamed for further analysis. 
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    return df
def splitting_data(df, col):
    '''
    Splitting data function takes in two arguments, one dataframe and the column you want to stratify on. After receiving these two arguments, the first split occurs and splits the data 60/40 for the train portion of the returns. A second split is performed on the remaining 40% of the data at a 50/50 ratio to make sure the validate and test returns are equal size. Random state of 24 because Kobe. Returns all 3 data sets.
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
   
    telco_train['monthly_charges'] = telco_train['monthly_charges'].astype(int)
    telco_val['monthly_charges'] = telco_val['monthly_charges'].astype(int)
    telco_test['monthly_charges'] = telco_test['monthly_charges'].astype(int)

    
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