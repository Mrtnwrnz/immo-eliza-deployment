import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import streamlit as st


@st.cache_data
def duplicates(df):
    """
    Check for, and remove duplicates, print out result, return modified DataFrame
    """
    if df.duplicated().any():
        rows = df.shape[0]
        df = df.drop_duplicates()
        print(f'{rows - df.shape[0]} duplicates removed')
    else:
        print('No duplicates found')
    return df

@st.cache_data
def remove_missing(df):
    """
    Remove all rows with 'MISSING' value, print out result, return modified DataFrame
    """
    rows_orig = df.shape[0]
    for i in df.columns:
        rows = df.shape[0]
        df = df[df[i] != 'MISSING']
        if (rows - df.shape[0]) > 0:
            print(f'For column ', i, ': ', rows - df.shape[0], ' rows containing "MISSING" were removed')
    print(f'TOTAL rows containing "MISSING" removed: ', rows_orig - df.shape[0], '\n')
    return df

@st.cache_data
def remove_empty(df):
    """
    Remove all rows with empty value, print out result, return modified DataFrame
    """
    rows_orig = df.shape[0]
    for i in df.columns:
        if i == 'surface_land_sqm':
            continue
        rows = df.shape[0]
        df = df[df[i].notnull()]
        if (rows - df.shape[0]) > 0:
            print(f'For column ', i, ': ', rows - df.shape[0], ' rows containing empty values were removed')
    print(f'TOTAL rows containing empty values removed: ', rows_orig - df.shape[0], '\n')
    return df

def encode_categorical(df):
    """
    Find and encodes columns with categorical values, return modified DataFrame
    """
    ordinals = {'state_building': [['AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE']], 
           'province': [['West Flanders', 'East Flanders', 'Walloon Brabant', 'Brussels', 'Hainaut', 'Antwerp', 'Liège', 'Namur', 'Flemish Brabant', 'Limburg', 'Luxembourg']], 
           'equipped_kitchen': [['NOT_INSTALLED', 'USA_UNINSTALLED', 'INSTALLED', 'USA_INSTALLED', 'SEMI_EQUIPPED', 'USA_SEMI_EQUIPPED', 'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED']], 
           'epc': [['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']], 
           'heating_type': [['SOLAR', 'ELECTRIC', 'GAS', 'PELLET', 'WOOD', 'FUELOIL', 'CARBON']]}
    for column, categories in ordinals.items():
        if column in df:
            encoder = OrdinalEncoder(categories=categories, dtype=int)
            df[column] = encoder.fit_transform(df[[column]].to_numpy()) + 1
    return df

def normalize(df):
    """
    Find and encodes columns with float values excluding price, return modified DataFrame
    """
    floats = df.drop('price', axis=1).select_dtypes(include=['float', 'int']).columns
    scaler = MinMaxScaler()
    df[floats] = scaler.fit_transform(df[floats])
    return df