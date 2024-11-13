import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def display_variables(df):
    """
    Display empty, 'MISSING' and unique values for all features, for pre- and post-cleaning
    """
    all_counts = {}
    for i in df.columns:
        entries = len(df[i])
        unique = len(df[i].unique())
        empty_count = df[i].isnull().sum()
        missing_count = 0
        if df[i].apply(lambda x: isinstance(x, str)).all():
            missing_count = df[i].str.contains('MISSING').sum()
        all_counts[i] = {'all': entries, 'unique': unique, 'empty': empty_count, 'missing': missing_count}
    bad_values = pd.DataFrame(all_counts).T
    print(bad_values)

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
    floats = df.drop('price', axis=1).select_dtypes(include='float').columns
    scaler = MinMaxScaler()
    df[floats] = scaler.fit_transform(df[floats])
    return df

def fill_missing_values(df):
    """
    Fill missing values: mean for floats, mode for integers, mode for categorical
    """
    for column in df.columns:
        # Check if the column contains floats
        if df[column].dtype == 'float64':
            # Replace empty values with the mean
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        # Check if the column contains integers
        elif df[column].dtype == 'int64':
            # Replace empty values with the mode
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
        # Remaining columns are categorical
        else:
            # Replace 'MISSING' wit 'NaN', and fill with the mode
            mode_value = df[column].mode()[0]
            df[column].replace('MISSING', mode_value, inplace=True)
    return df