import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ✅ Ensure 'model/' directory exists
os.makedirs('model', exist_ok=True)

# ✅ Load Data from URL
url = 'https://raw.githubusercontent.com/ShaliniKashyap717/data/refs/heads/main/hotel_booking%20(1).csv'
df = pd.read_csv(url)

# ✅ Drop PII & leakage columns
df.drop(columns=['name', 'email', 'phone-number', 'credit_card'], inplace=True)
df.drop(columns=['reservation_status', 'reservation_status_date'], inplace=True)

# ✅ Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ✅ Split features and target
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
from sklearn.preprocessing import LabelEncoder

# Example encoding for one column
le = LabelEncoder()
X_train['hotel'] = le.fit_transform(X_train['hotel'])
X_test['hotel'] = le.transform(X_test['hotel'])


# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model and encoders to 'model/' folder
joblib.dump(model, 'hotel_cancellation_model.pkl,compress=9')
joblib.dump(label_encoders, 'label_encoders.pkl')
df.to_csv('preprocessed_data.csv', index=False)

print("✅ Files saved in 'model/' folder:")
print("- hotel_cancellation_model.pkl")
print("- label_encoders.pkl")
print("- preprocessed_data.csv")
