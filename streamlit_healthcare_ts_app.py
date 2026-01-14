
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

st.title("Hospital Admissions Forecasting")

df = pd.read_csv("hospital_daily_admissions.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

st.subheader("Daily Admissions Data")
st.line_chart(df)

p = st.slider("AR order (p)", 0, 5, 2)
d = st.slider("Differencing (d)", 0, 2, 1)
q = st.slider("MA order (q)", 0, 5, 2)

model = ARIMA(df["admissions"], order=(p,d,q))
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)

st.subheader("30-Day Forecast")
st.line_chart(forecast)
