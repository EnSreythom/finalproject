import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import mlflow

# Preprocess data
def preprocess_data(file_path):
    column_names = ['serial', 'date', 'age', 'distance', 'stores', 'latitude', 'longitude', 'price']
    df = pd.read_csv(file_path, names=column_names)
    df = df.iloc[:, 1:]  # Drop the first column
    df_norm = (df - df.mean()) / df.std()  # Normalize the data
    return df_norm

# Define the model
def get_model():
    model = Sequential([
        Dense(10, input_shape=(5,), activation='relu'),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Main function
if __name__ == "__main__":
    df_norm = preprocess_data('data.csv')
    
    x = df_norm.iloc[:, :-1].values
    y = df_norm.iloc[:, -1].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

    mlflow.start_run()
    model = get_model()
    es_cb = EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[es_cb])
    
    mlflow.log_metric("final_loss", history.history['loss'][-1])
    mlflow.end_run()