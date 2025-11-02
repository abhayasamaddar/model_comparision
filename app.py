import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

st.set_page_config(page_title="Air Quality Model Comparison", layout="wide")

# --------------------- Supabase Setup ---------------------
SUPABASE_URL = st.secrets["https://fjfmgndbiespptmsnrff.supabase.co"]
SUPABASE_KEY = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"]

@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=300)
def load_data():
    supabase = init_supabase()
    response = supabase.table('airquality').select('*').execute()
    if not response.data:
        st.error("No data found in Supabase table.")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')

    numeric_cols = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
    return df

def create_features(df, target_columns, n_lags=3):
    df_eng = df.copy()
    safe_n_lags = min(n_lags, len(df_eng) - 1)
    for col in target_columns:
        for lag in range(1, safe_n_lags + 1):
            df_eng[f"{col}_lag_{lag}"] = df_eng[col].shift(lag)
    df_eng = df_eng.dropna().reset_index(drop=True)
    return df_eng

# --------------------- Streamlit App ---------------------
st.title("🌍 Air Quality Prediction — Model Comparison")

with st.spinner("Loading data..."):
    df = load_data()

if df.empty:
    st.stop()

# Use only last 2000 samples
if len(df) > 2000:
    df = df.tail(2000).reset_index(drop=True)
    st.info(f"Using last 2000 samples for training. ({len(df)} samples)")
else:
    st.info(f"Using all {len(df)} samples for training.")

st.write("### Data Preview")
st.dataframe(df.tail(10))

target_columns = ['pm25', 'pm10', 'co2', 'co', 'temperature', 'humidity']
df_eng = create_features(df, target_columns)
features = [c for c in df_eng.columns if c not in target_columns + ['id', 'created_at']]

X = df_eng[features].values
y = df_eng[target_columns].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ Train Models ------------------
models = {
    "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)),
    "LSTM": None  # LSTM handled separately
}

st.subheader("Training Traditional Models")
results = {}

for name, model in models.items():
    if model is None:
        continue
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    results[name] = {"RMSE": rmse, "R²": r2}
    st.success(f"{name}: RMSE={rmse:.3f}, R²={r2:.3f}")

# ------------------ LSTM ------------------
st.subheader("Training LSTM Model")
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

model_lstm = Sequential([
    LSTM(64, input_shape=(1, X_scaled.shape[1]), return_sequences=False),
    Dropout(0.2),
    Dense(y.shape[1])
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y, epochs=10, batch_size=16, verbose=0)
pred_lstm = model_lstm.predict(X_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y, pred_lstm))
r2_lstm = r2_score(y, pred_lstm)
results["LSTM"] = {"RMSE": rmse_lstm, "R²": r2_lstm}
st.success(f"LSTM: RMSE={rmse_lstm:.3f}, R²={r2_lstm:.3f}")

# ------------------ Plot Comparison ------------------
st.subheader("📊 Model Performance Comparison")
fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "R²"))
names = list(results.keys())
rmse_vals = [results[m]["RMSE"] for m in names]
r2_vals = [results[m]["R²"] for m in names]

fig.add_trace(go.Bar(x=names, y=rmse_vals, name="RMSE"), 1, 1)
fig.add_trace(go.Bar(x=names, y=r2_vals, name="R²"), 1, 2)
fig.update_layout(height=400, width=900, showlegend=False)
st.plotly_chart(fig)

st.success("✅ Training complete and models compared successfully!")
