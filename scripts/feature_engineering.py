import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    """Prepares features for the model."""
    # Convert categorical data to numeric using LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select relevant features
    selected_features = ['Severity', 'Distance(mi)', 'Temperature(F)', 'Visibility(mi)', 'Humidity(%)']

    df = df[selected_features]

    return df, label_encoders

if __name__ == "__main__":
    
    df = pd.read_csv('./data/US_Accidents_Cleaned.csv')
    print("Columns in dataset:", df.columns)
    df_features, encoders = feature_engineering(df)
    df_features.to_csv('./data/US_Accidents_Features.csv', index=False)
    print("Feature engineering completed and saved!")
