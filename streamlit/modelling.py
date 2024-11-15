from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

@st.cache_data
def apply_regressor(df, house_apt, enc_name):
    """
    Split data, apply regression model to the DataFrame, print out model scores and return fitted encoder
    Arguments: df (DataFrame); house_apt (house, apartment); enc_name (linear, lasso, randomforest, xgboost)
    """
    # define encoders
    encoders = {'linear': LinearRegression(), 'lasso': Lasso(), 'randomforest': RandomForestRegressor(n_estimators=12, random_state=42), 'xgboost': XGBRegressor(learning_rate=0.01, n_estimators=1050, max_depth=5, subsample=0.6, colsample_bytree=0.5, random_state=42)}
    # state type of data used
    if house_apt not in ['apartment', 'house', 'full']:
        return print('Please input "house", "apartment" of "full" as a second argument to call this function')
    # split dataset
    X, y = np.array(df.drop(columns='price')), np.array(df['price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    # initialize and train model
    enc = encoders[enc_name]
    enc.fit(X_train, y_train)
    
    return enc