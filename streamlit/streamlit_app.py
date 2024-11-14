from clean_preprocess import *
from modelling import *
import streamlit as st

st.title("Immo Eliza's price predictor")

# import csv from scraping
df = pd.read_csv("..\data\properties.csv")

# examine features: rows, unique values, count empty/'MISSING' values
display_variables(df)

# check for duplicates
df = duplicates(df)

# Define limitations of 'standard' house
df = df[df['price'] < 1000000]
df = df[df['nbr_bedrooms']< 6]
df = df[df['subproperty_type'] != 'CASTLE']

# Drop features without use
df = df.drop(['id', 'region', 'locality', 'zip_code', 'latitude', 'longitude', 'subproperty_type'], axis='columns')
# Drop features to prevent overfitting / correlatetion
df = df.drop(['fl_terrace', 'nbr_bedrooms', 'equipped_kitchen', 'epc', 'fl_double_glazing', 'construction_year', 'heating_type', 'fl_furnished'], axis='columns')

# clean data
df_no_missing = remove_missing(df)
df_no_empty = remove_empty(df_no_missing)

# examine features after cleaning
display_variables(df_no_empty) 

# Encode and normalize
df_encoded = encode_categorical(df_no_empty)
df_clean = normalize(df_encoded)


# create seperate datasets: APARTMENT / HOUSE
df_house = df_clean[df_clean['property_type'] == 'HOUSE'].drop(columns=['property_type', 'garden_sqm'])
df_apartment = df_clean[df_clean['property_type'] == 'APARTMENT'].drop(columns=['property_type', 'surface_land_sqm'])

# create model and save it

save_model(apply_regressor(df_house, 'house', 'xgboost'), 'house')
save_model(apply_regressor(df_apartment, 'apartment', 'linear'), 'apartment')



# load model
#loaded_model_house = joblib.load('../data/model_file_house.pkl')
#loaded_model_apartment = joblib.load('../data/model_file_apartment.pkl')
