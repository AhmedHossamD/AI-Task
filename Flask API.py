import numpy as np
from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model and the scaler
model_path = 'rf_model.pkl'  
scaler_path = 'scaler.pkl'   

# Try to load the model and scaler, print messages if successful
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        print(f"Received data: {data}")  # Debug: Print received data
        
        # Extract features from the input JSON
        features = [
            data['Square_Feet'],
            data['Bedrooms'],
            data['Age'],
            data['Location_Rating']
        ]
        print(f"Features: {features}")  # Debug: Print extracted features
        
        # Convert the input into a numpy array and reshape it for the model
        features_array = np.array(features).reshape(1, -1)
        print(f"Features array: {features_array}")  # Debug: Print features array
        
        # Scale the input features
        scaled_features = scaler.transform(features_array)
        print(f"Scaled features: {scaled_features}")  # Debug: Print scaled features
        
        # Make the prediction using the loaded model
        prediction = model.predict(scaled_features)
        print(f"Prediction: {prediction[0]}")  # Debug: Print prediction
        
        # Return the prediction as JSON
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        print(f"Error: {e}")  # Print the error for debugging purposes
        return jsonify({'error': str(e)})

# Main entry point to run the Flask app
if __name__ == '_main_':
    app.run(debug=True)