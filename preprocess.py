import pandas as pd

def preprocess_data(file_path):
    column_names = ['serial', 'date', 'age', 'distance', 'stores', 'latitude', 'longitude', 'price']
    df = pd.read_csv(file_path, names=column_names)
    df = df.iloc[:, 1:]  # Drop the first column
    df_norm = (df - df.mean()) / df.std()  # Normalize the data
    return df_norm

if __name__ == "__main__":
    # Specify the full path to your data file
    file_path = r'D:\Final project\data.csv'  # Use raw string for the path
    df_normalized = preprocess_data(file_path)
    df_normalized.to_csv('data_normalized.csv', index=False)