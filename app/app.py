from flask import Flask, request, jsonify
import numpy as np
import joblib

class BrentOilModelAPI:
    def __init__(self, model_path, scaler_path):
        self.app = Flask(__name__)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Define the routes
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])

    def predict(self):
        data = request.json  # Get data from the request
        input_data = np.array(data['input']).reshape(-1, 1)  # Adjust based on your model's input shape
        input_data_scaled = self.scaler.transform(input_data)

        # Make prediction
        prediction = self.model.predict(input_data_scaled)
        prediction = self.scaler.inverse_transform(prediction)  # Inverse transform if necessary

        return jsonify({'prediction': prediction.tolist()})

    def run(self, debug=True):
        self.app.run(debug=debug)

if __name__ == '__main__':
    model_path = 'path_to_your_model.pkl'  # Adjust the path accordingly
    scaler_path = 'path_to_your_scaler.pkl'  # If you used a scaler
    api = BrentOilModelAPI(model_path, scaler_path)
    api.run()
