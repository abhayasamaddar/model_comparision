import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 🔐 Connect to Supabase
# -------------------------------
SUPABASE_URL = st.secrets["https://fjfmgndbiespptmsnrff.supabase.co"]
SUPABASE_KEY = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("🌍 Air Quality Prediction Dashboard")
st.write("Comparison of **Random Forest**, **XGBoost**, and **LSTM** models using the last 2000 samples.")

# -------------------------------
# 📥 Load Data
# -------------------------------
@st.cache_data
def load_data():
    data = supabase.table("airquality").select("*").execute()
    df = pd.DataFrame(data.data)
    return df

df = load_data()

if df.empty:
    st.error("No data found in Supabase table 'airquality'.")
    st.stop()

# -------------------------------
# 🧹 Preprocess
# -------------------------------
df = df.sort_values(by=df.columns[0])  # sort by timestamp/ID if present
df = df.tail(2000)  # use last 2000 samples only

# Drop non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.empty:
    st.error("No numeric columns found for model training.")
    st.stop()

X = numeric_df.drop(columns=numeric_df.columns[-1])  # all except target
y = numeric_df[numeric_df.columns[-1]]  # target variable

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# -------------------------------
# 🌳 Random Forest
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
rf_r2 = r2_score(y_test, rf_pred)

# -------------------------------
# ⚡ XGBoost
# -------------------------------
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
xgb_r2 = r2_score(y_test, xgb_pred)

# -------------------------------
# 🔁 LSTM
# -------------------------------
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
X_train_lstm, X_test_lstm = X_lstm[:int(0.8 * len(X_lstm))], X_lstm[int(0.8 * len(X_lstm)):]
y_train_lstm, y_test_lstm = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=32, verbose=0, callbacks=[early_stop])

lstm_pred = lstm_model.predict(X_test_lstm).flatten()
lstm_rmse = mean_squared_error(y_test_lstm, lstm_pred, squared=False)
lstm_r2 = r2_score(y_test_lstm, lstm_pred)

# -------------------------------
# 📊 Show Results
# -------------------------------
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LSTM'],
    'RMSE': [rf_rmse, xgb_rmse, lstm_rmse],
    'R² Score': [rf_r2, xgb_r2, lstm_r2]
})

st.subheader("📈 Model Comparison Metrics")
st.dataframe(results)

# Plot comparison
fig = px.bar(results, x='Model', y='RMSE', color='Model', title='Model RMSE Comparison', text_auto='.2f')
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 🔍 Detailed Plots
# -------------------------------
st.subheader("🔍 Model Predictions")

def plot_predictions(y_true, y_pred, title):
    df_plot = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
    fig = px.line(df_plot, title=title)
    st.plotly_chart(fig, use_container_width=True)

plot_predictions(y_test, rf_pred, "Random Forest Prediction vs True")
plot_predictions(y_test, xgb_pred, "XGBoost Prediction vs True")
plot_predictions(y_test_lstm, lstm_pred, "LSTM Prediction vs True")

st.success("✅ Model training and comparison complete using last 2000 samples!")
