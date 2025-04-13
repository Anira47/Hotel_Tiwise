import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, encoders, and data
model = joblib.load('model/hotel_cancellation_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')
df = pd.read_csv('model/preprocessed_data.csv')

st.title("🏨 Hotel Cancellation Risk Predictor")

# 🔹 Hotel Selection
hotel_le = label_encoders['hotel']
hotels = hotel_le.inverse_transform(sorted(df['hotel'].unique()))
selected_hotel = st.selectbox("Select a Hotel", hotels)

if selected_hotel:
    # Encode selected hotel
    hotel_encoded = hotel_le.transform([selected_hotel])[0]
    
    # Filter data for that hotel
    hotel_df = df[df['hotel'] == hotel_encoded].copy()
    
    if hotel_df.empty:
        st.warning("No data found for the selected hotel.")
    else:
        # Drop target column and make predictions
        X = hotel_df.drop(columns=['is_canceled'])
        hotel_df['predicted_risk'] = model.predict_proba(X)[:, 1]

        # Group by lead_time and average predicted risk
        grouped = hotel_df.groupby('lead_time')['predicted_risk'].mean().reset_index()

        # 🔹 Plot chart
        st.subheader("📊 Cancellation Risk vs. Lead Time")
        fig, ax = plt.subplots()
        ax.plot(grouped['lead_time'], grouped['predicted_risk'], marker='o')
        ax.set_xlabel("Lead Time (Days)")
        ax.set_ylabel("Predicted Cancellation Risk")
        ax.set_title(f"Cancellation Risk for {selected_hotel}")
        ax.grid(True)
        st.pyplot(fig)
