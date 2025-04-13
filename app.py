from flask import Flask, request, jsonify
import pandas as pd
import joblib

from flask_cors import CORS  # Allow frontend requests

app = Flask(__name__)
print("🚀 Starting Flask app...")

CORS(app)  # Enable CORS

# Load model and encoders
model = joblib.load('model/hotel_cancellation_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')
df = pd.read_csv('model/preprocessed_data.csv')


# 🔹 Get available hotels
@app.route('/hotels', methods=['GET'])
def get_hotels():
    hotel_le = label_encoders['hotel']
    hotels = hotel_le.inverse_transform(sorted(df['hotel'].unique()))
    return jsonify({'hotels': list(hotels)})

# 🔹 Get chart data for selected hotel
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_hotel = data.get('hotel')

    if not selected_hotel:
        return jsonify({'error': 'Hotel not provided'}), 400

    # Encode hotel name
    hotel_encoded = label_encoders['hotel'].transform([selected_hotel])[0]
    
    # Filter data for that hotel
    hotel_df = df[df['hotel'] == hotel_encoded].copy()

    if hotel_df.empty:
        return jsonify({'error': 'No data found for selected hotel'}), 404

    # Predict cancellation risk
    X = hotel_df.drop(columns=['is_canceled'])
    hotel_df['predicted_risk'] = model.predict_proba(X)[:, 1]

    # Group by lead_time and average the predicted risk
    grouped = hotel_df.groupby('lead_time')['predicted_risk'].mean().reset_index()

    # Return chart data
    return jsonify({
        'x': grouped['lead_time'].tolist(),
        'y': grouped['predicted_risk'].round(3).tolist()
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
