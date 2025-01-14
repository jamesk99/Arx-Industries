import os
import yfinance as yf
import pandas as pd
import streamlit as st

def fetch_and_save_data():
    ticker = st.sidebar.text_input("Enter ticker symbol:")
    # Create a unique filename based on the ticker symbol
    filename = f"{ticker}_stock_data.csv"
    
    # Fetch historical data for the ticker
    data = yf.download(ticker, period="1y", interval="1d")
    
    if data.empty:
        print(f"No data found for ticker '{ticker}'.")
        return

    # Check if the file exists
    if os.path.exists(filename):
        # Load the existing data
        existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # Update the data by appending new data, avoiding duplicates
        combined_data = pd.concat([existing_data, data]).drop_duplicates()
        combined_data.sort_index(inplace=True)  # Sort by date

        # Save the updated data to the same CSV
        combined_data.to_csv(filename)
        print(f"Data for {ticker} updated in {filename}.")
    else:
        # If file does not exist, save the new data to CSV
        data.to_csv(filename)
        print(f"Data for {ticker} saved to {filename}.")
        
fetch_and_save_data()
