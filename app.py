from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('path_to_your_trained_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assuming data is a list of features
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': prediction[0][0].tolist()})

if __name__ == "__main__":
    app.run(debug=True)