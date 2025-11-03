from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model once on server start
with open('XGBoost_best_model.pkl', 'rb') as f:  # Replace with your actual model file
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from JSON request - ensure keys match frontend fields
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], 
            data['fbs'], data['restecg'], data['thalach'], data['exang'], data['oldpeak'], 
            data['slope'], data['ca'], data['thal']
        ]

        # Convert to numpy array with shape (1, n_features)
        input_array = np.array(features).reshape(1, -1)

        # Make prediction
        pred_class = model.predict(input_array)[0]
        pred_prob = model.predict_proba(input_array)[0][1]  # Probability of positive class

        response = {
            'prediction': int(pred_class),
            'probability': float(pred_prob)
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
