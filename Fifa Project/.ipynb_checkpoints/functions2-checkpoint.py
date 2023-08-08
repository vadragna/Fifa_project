def clean_data(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from functions2 import clean_data
    data2 = df.drop(["ID", "Name", "Nationality", "Club", "Position", "Joined", "Loan Date End", "Wage" , "Release Clause", "GK Diving", "GK Handling" , "GK Kicking", "GK Positioning", "GK Reflexes"], axis = 1)
    def feet_to_cm(height_str):
        feet, inches = map(int, height_str.replace('"', '').split("'"))
        total_inches = (feet * 12) + inches
        cm = total_inches * 2.54
        return cm

    data2['Height'] = data2['Height'].apply(feet_to_cm)
    data2['Weight'] = data2['Weight'].str.replace('lbs', '').astype(float)
    def convert_value(value_str):
        value_str = value_str.replace('€', '') 
        if value_str.endswith('K'):
            return float(value_str[:-1]) * 1000  
        elif value_str.endswith('M'):
            return float(value_str[:-1]) * 1000000  
        else:
            return float(value_str) 
            
    data2['Value'] = data2['Value'].apply(convert_value)
    data2["W/F"] = data2["W/F"].str.replace("★", "" )
    data2["W/F"] = pd.to_numeric(data2["W/F"], errors='coerce')
    data2["SM"] = data2["SM"].str.replace("★", "" )
    data2["SM"] = pd.to_numeric(data2["SM"], errors='coerce')
    data2["IR"] = data2["IR"].str.replace(" ★", "" )
    data2["IR"] = pd.to_numeric(data2["IR"], errors='coerce')
    data2["D/W"] = data2["D/W"].fillna('Medium') 
    data2["A/W"] = data2["D/W"].fillna('Medium')
    categorical = data2.select_dtypes('object')
    numerical = data2._get_numeric_data()
    categorical['A/W'].value_counts(dropna=False)
    categorical['D/W'].value_counts(dropna=False)
    X = numerical.drop(["OVA"], axis = 1)
    y = data2["OVA"]
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_df, columns=X.columns)
    encoder = OneHotEncoder(drop='first').fit(categorical)

    cols = encoder.get_feature_names_out(input_features=categorical.columns)
    categorical_encode = pd.DataFrame(encoder.transform(categorical).toarray(),columns=cols)
    data3 = pd.concat([scaled_df, categorical_encode, y], axis=1)
    return data3


def only_clean(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from functions2 import clean_data
    df = df[['BP', 'Age', 'Height', 'Weight', 'foot', 'Growth', 'Value', 'Skill', 'Movement', 'Power', 'Mentality', 'Defending', 'Goalkeeping', 'Total Stats', 'W/F', 'SM', 'A/W', 'D/W', 'IR', 'PAC', 'SHO', 'PAS','DRI', 'DEF', 'PHY','OVA']]
    data2 = df
    def feet_to_cm(height_str):
        feet, inches = map(int, height_str.replace('"', '').split("'"))
        total_inches = (feet * 12) + inches
        cm = total_inches * 2.54
        return cm

    data2['Height'] = data2['Height'].apply(feet_to_cm)
    data2['Weight'] = data2['Weight'].str.replace('lbs', '').astype(float)
    def convert_value(value_str):
        value_str = value_str.replace('€', '') 
        if value_str.endswith('K'):
            return float(value_str[:-1]) * 1000  
        elif value_str.endswith('M'):
            return float(value_str[:-1]) * 1000000  
        else:
            return float(value_str)             
    data2['Value'] = data2['Value'].apply(convert_value)
    data2["W/F"] = data2["W/F"].str.replace("★", "" )
    data2["W/F"] = pd.to_numeric(data2["W/F"], errors='coerce')
    data2["SM"] = data2["SM"].str.replace("★", "" )
    data2["SM"] = pd.to_numeric(data2["SM"], errors='coerce')
    data2["IR"] = data2["IR"].str.replace(" ★", "" )
    data2["IR"] = pd.to_numeric(data2["IR"], errors='coerce')
    data2["D/W"] = data2["D/W"].fillna('Medium') 
    data2["A/W"] = data2["D/W"].fillna('Medium')
    return data2
    
    
    
def min_max_encode(df,z):
    
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    y = df[z]
    categorical = df.select_dtypes('object')
    numerical = df._get_numeric_data().drop([z], axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(numerical)
    numerical_scaled = transformer.transform(numerical)
    numerical_scaled = pd.DataFrame(numerical_scaled, columns=numerical.columns)
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first').fit(categorical)
    cols = encoder.get_feature_names_out(input_features=categorical.columns)
    categorical_encode = pd.DataFrame(encoder.transform(categorical).toarray(),columns=cols)
    
    X = pd.concat([numerical_scaled, categorical_encode], axis=1)
    
    return y, X