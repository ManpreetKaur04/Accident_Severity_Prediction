import pandas as pd

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path, on_bad_lines='skip')

def clean_data(df):
    """Performs data cleaning."""
    # Drop columns with more than 30% missing values
    threshold = 0.3 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    
    # Fill numeric columns with median and categorical with mode
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

if __name__ == "__main__":
    file_path = "your_file_path"
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    df_cleaned.to_csv("./data/US_Accidents_Cleaned.csv", index=False)
    print("Data cleaning completed and saved!")
