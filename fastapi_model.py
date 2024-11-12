from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd
import joblib 
from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI()

class Listing(BaseModel):
    property_type: str
    province: str
    total_area_sqm: float
    surface_land_sqm: float
    nbr_frontages: int
    fl_open_fire: int
    terrace_sqm: float
    fl_garden: int
    garden_sqm: float
    fl_swimming_pool: int
    fl_floodzone: int
    state_building: str
    primary_energy_consumption_sqm: float
    cadastral_income: float


@app.post("/prediction/")
def encode(item: Listing):
    df = pd.DataFrame([item.dict()])
    # encode ordinals, scale floats
    ordinals = {'state_building': [['AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP', 'TO_RENOVATE', 'TO_RESTORE']], 
                'province': [['West Flanders', 'East Flanders', 'Walloon Brabant', 'Brussels', 'Hainaut', 'Antwerp', 'Liège', 'Namur', 'Flemish Brabant', 'Limburg', 'Luxembourg']]}
    for column, categories in ordinals.items():
        if column in df:
            encoder = OrdinalEncoder(categories=categories, dtype=int)
            df[column] = encoder.fit_transform(df[[column]].to_numpy()) + 1
    floats = df.select_dtypes(include='float').columns
    scaler = MinMaxScaler()
    df[floats] = scaler.fit_transform(df[floats])

    # select house/apartment, load model
    if item.property_type == 'HOUSE':
        loaded_model = joblib.load('models/model_immo_eliza_house.pkl')
    elif item.property_type == 'APARTMENT':
        loaded_model = joblib.load('models/model_immo_eliza_apartment.pkl')
    else:
        return {"error": "Invalid property type. Must be 'HOUSE' or 'APARTMENT'."}
    
    # predict price
    if item.property_type == 'HOUSE':
        prediction = loaded_model.predict(df.drop(columns= ['property_type', 'garden_sqm']))
    elif item.property_type == 'APARTMENT':
        prediction = loaded_model.predict(df.drop(columns= ['property_type', 'surface_land_sqm']))
    return {"predicted_price": float(prediction[0])}
