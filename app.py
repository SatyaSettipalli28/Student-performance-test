import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title
st.title("🎓 Student Performance Analysis")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features
features = ['reading score', 'writing score']
target = 'math score'

X = df[features]
y = df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# User input
st.subheader("📊 Predict Math Score")
reading = st.slider("Reading Score", 0, 100, 50)
writing = st.slider("Writing Score", 0, 100, 50)

# Prediction
prediction = model.predict([[reading, writing]])

st.success(f"Predicted Math Score: {prediction[0]:.2f}")
