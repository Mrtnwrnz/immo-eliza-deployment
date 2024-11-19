from clean_preprocess import *
from modelling import *
import streamlit as st
import pandas as pd

st.title("Immo Eliza's price predictor")

# import csv from scraping
df = pd.read_csv("data/properties.csv")

# check for duplicates
df = duplicates(df)
st.write(df.shape)
# Define limitations of 'standard' house
df = df[df['price'] < 1000000]
df = df[df['nbr_bedrooms']< 6]
df = df[df['subproperty_type'] != 'CASTLE']
st.write(df.shape)
# Drop features without use
df = df.drop(['id', 'region', 'locality', 'zip_code', 'latitude', 'longitude', 'subproperty_type'], axis='columns')
# Drop features to prevent overfitting / correlatetion
df = df.drop(['terrace_sqm', 'nbr_bedrooms', 'equipped_kitchen', 'primary_energy_consumption_sqm', 
              'fl_double_glazing', 'construction_year', 'heating_type', 'fl_furnished'], axis='columns')
st.write(df.shape)
# clean data
df_no_missing = remove_missing(df)
df_no_empty = remove_empty(df_no_missing)
st.write(df.shape)
# Encode and normalize
df_encoded = encode_categorical(df_no_empty)
df_clean = normalize(df_encoded)
st.write(df.shape)
# create seperate datasets: APARTMENT / HOUSE
df_house = df_clean[df_clean['property_type'] == 'HOUSE'].drop(columns=['property_type', 'garden_sqm'])
df_apartment = df_clean[df_clean['property_type'] == 'APARTMENT'].drop(columns=['property_type', 'surface_land_sqm'])

# create model
model_house= apply_regressor(df_house, 'house', 'xgboost')
model_apt = apply_regressor(df_apartment, 'apartment', 'xgboost')

st.write('Please enter your property details below')

# input data for prediction
col1, col2 = st.columns(2)
with col1:
    colA, colB = st.columns(2, gap="medium")
    with colA:
        property_type = st.segmented_control('Property type', ('House', 'Apartment'))
    with colB:
        fl_floodzone = 1 if st.segmented_control('Floodzone', ('Yes', 'No')) == 'Yes' else 0
    total_area_sqm = st.slider('Total living area', 25, 500, 100)
    nbr_frontages = int(st.select_slider('Number of frontages', ['1', '2', '3', '4 (or more)']).split()[0])
    epc = st.select_slider('Select your EPC', ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
    cadastral_income = st.slider('Cadastral income', 500, 50000, 15000)  
with col2:
    province = st.selectbox('Select your province', ('West Flanders', 'East Flanders', 'Walloon Brabant', 
                                                     'Brussels', 'Hainaut', 'Antwerp', 'Liège', 'Namur', 
                                                     'Flemish Brabant', 'Limburg', 'Luxembourg'))
    colA, colB = st.columns(2)
    with colA:
        fl_terrace = 1 if st.checkbox('Terrace') else 0
        fl_garden = 1 if st.checkbox('Garden') else 0
    with colB:
        fl_open_fire = 1 if st.checkbox('Open fire') else 0
        fl_swimming_pool = 1 if st.checkbox('Swimming pool') else 0
    state_building = st.selectbox('State of the property', ('AS_NEW', 'JUST_RENOVATED', 'GOOD', 
                                                            'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE'))
    garden_sqm = st.slider('Garden surface', 0, 5000)
    surface_land = st.slider('Surface lot', 25, 50000, 200)


input_data = pd.DataFrame([{
    'province': province, 'total_area_sqm': total_area_sqm, 'nbr_frontages': nbr_frontages,
'fl_open_fire': fl_open_fire, 'fl_terrace': fl_terrace, 'fl_swimming_pool': fl_swimming_pool, 
'fl_floodzone': fl_floodzone, 'state_building': state_building, 'cadastral_income': cadastral_income, 
'fl_garden': fl_garden, 'epc': epc, 'price': 0
}])

if property_type == 'House':
    input_data['surface_land_sqm'] = surface_land
elif property_type == 'Apartment':
    input_data['garden_sqm'] = garden_sqm

encoded = encode_categorical(input_data)
normalized = normalize(encoded)
input_clean = normalized.drop('price', axis=1)

# predict
result = ''
if property_type == 'House':
    result = model_house.predict(input_clean)
elif property_type == 'Apartment':
    result = model_apt.predict(input_clean)


st.write(f'The predicted value for your property is {result} EUR')