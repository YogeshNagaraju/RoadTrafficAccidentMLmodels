import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier               
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
@st.cache
def load_data():
    return pd.read_csv('RTA Dataset.csv')

data = load_data()

# Preprocess Data
data.dropna(subset=['Accident_severity'], inplace=True)
data.fillna(method='ffill', inplace=True)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('Accident_severity', axis=1)
y = data['Accident_severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)

# Streamlit Frontend
st.title('Smart Traffic Management: Accident Severity Prediction')
st.sidebar.header('Input Features')

# Input Features
def user_input_features():
    inputs = {}
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            inputs[col] = st.sidebar.selectbox(f'{col}', options)
        else:
            inputs[col] = st.sidebar.number_input(f'{col}', min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
    return pd.DataFrame([inputs])

user_data = user_input_features()

# Encode User Input
for col, le in label_encoders.items():
    if col in user_data:
        user_data[col] = le.transform(user_data[col])

# Scale User Input
user_data_scaled = scaler.transform(user_data)

# Predictions
if st.sidebar.button('Predict Analysis'):
    st.subheader('Prediction Results')
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(user_data_scaled)[0]
        decoded_pred = label_encoders['Accident_severity'].inverse_transform([pred])[0]
        predictions[model_name] = decoded_pred
        st.write(f'{model_name}: {decoded_pred}')

    # Model Comparison
    st.subheader('Model Comparison')
    comparison = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        comparison[model_name] = accuracy

    comparison_df = pd.DataFrame(list(comparison.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    st.dataframe(comparison_df)
