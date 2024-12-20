import streamlit as st
import pandas as pd
import yfinance as yf

st.write("""
         # Simple Stock Price App
         Shown are the stock **closing price** and ***volume*** of Apple! 
         """)

tickersymbol = "AAPL"
tickerData = yf.Ticker(tickersymbol)
tickerDf = tickerData.history(period = "1d", start = "2024-1-2", end = "2024-12-1")

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)