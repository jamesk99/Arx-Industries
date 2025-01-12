import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from math import log, sqrt, exp
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(layout="wide")  # Make the app full width

def format_number(x):
    """Format numbers to be more readable"""
    if isinstance(x, float):
        if abs(x) > 1:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return x

def black_scholes_put_price(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put_price = (K * exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price

def line(x, y, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig

def main():
    st.title("Tail-Risk Hedging Strategy Dashboard")
    
    # Create two columns for the sidebar parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        ticker = st.text_input("Ticker Symbol:", value="SPY")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-08"))
        
        stock_allocation = st.slider("Equity Allocation (%)", 
                            min_value=0.0, max_value=100.0, value=90.0)
        put_allocation = 100.0 - stock_allocation
        
        strike_pct = st.slider("Strike % OTM", 
                            min_value=50.0, max_value=100.0, value=80.0, help= "80 means strike is 20% below current price")
        
        years_to_expiry = st.slider("Option Expiry (Years)", 
                            min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        
        risk_free_rate = st.slider("Risk-Free Rate (%)", 
                            min_value=0.0, max_value=8.0, value=3.0, step=0.1) / 100.0
        
        vol_override = st.slider("Vol Override (%)", 
                            min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0

        initial_price = st.text_input("Initial Price", value=400.0, max_chars=7)
        initial_price = float(initial_price)    
    
    with col2:
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)
                                             
            if data.empty:
                st.error(f"No data found for {ticker}")
                st.stop()
                
            # Format the display data
            display_df = data.copy()
            for col in display_df.columns:
                display_df[col] = display_df[col].apply(format_number)

            # Show data sample
            st.subheader("Market Data")
            st.dataframe(
                display_df.tail(10),
                use_container_width=True,
                hide_index=False
            )
            
            # Calculate returns and volatility
            data['returns'] = data['Close'].pct_change(periods=7)
            data = data.dropna()
            
            current_price = float(data['Close'].iloc[-1])
            strike_price = current_price * (strike_pct / 100.0)
            
            data["stock_value"] = (1+data['returns']).cumprod()

            # Calculate volatility
            daily_vol = float(data['returns'].std())
            ann_vol_est = daily_vol * np.sqrt(252)
            ann_vol = vol_override if vol_override > 0 else ann_vol_est
            
            # Calculate put price
            put_price = black_scholes_put_price(
                S=current_price,
                K=strike_price,
                T=years_to_expiry,
                r=risk_free_rate,
                sigma=ann_vol
            )
            
            # Display key metrics
            st.subheader("Key Metrics")
            cols = st.columns(4)
            
            # Format values as strings before passing to metric
            cols[0].metric("Current Price", f"${current_price:.2f}")
            cols[1].metric("Strike Price", f"${strike_price:.2f}")
            cols[2].metric("Put Price", f"${put_price:.2f}")
            cols[3].metric("Ann. Volatility", f"{ann_vol_est:.1%}")
                       
            # Price chart
            st.subheader("Price History")
            df_price = data.reset_index()[['Date', 'Close']]  # Reset index to get 'Date' as a column
            
                    #Now df_price is a DataFrame with columns 'Date' and 'Close'
                    # If your index was named 'Date' after reset_index, rename columns if needed:
                    # df_price.columns = ['Date', 'Close']

            fig_price = px.line(data_frame=df_price,
                x='Date', 
                y='Close',
                x_title="Time",
                y_title="Price",
                title=f"{ticker} Price History",
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
                                   
            # Portfolio value chart
            st.subheader("Portfolio Value (Starting at $1)")
            df_value = data.reset_index()[['Date', 'stock_value']]         # So that date will become a column
            fig_value = px.line(
                df_value,
                x='Date',   # or whatever your index name becomes
                y='stock_value',
                x_title="Time",
                y_title="Account Value",
                title="Portfolio Value Over Time",
                height=400
            )
            st.plotly_chart(fig_value, use_container_width=True)
            
                       
            # Calculate scenario analysis
            final_price = float(data['Close'].iloc[-10].item()) #Using .item() to get scalar value
            put_payoff = max(strike_price - final_price, 0.0)
            
            put_price = max(put_price, 0.01) # Floor put price
            num_puts = (put_allocation / 100.0) / put_price
            
            final_stock_value = (stock_allocation / 100.0) * (final_price / initial_price)
            final_put_value = num_puts * put_payoff
            total_value = final_stock_value + final_put_value
            
            # Display scenario analysis
            st.subheader("Scenario Analysis")
            cols = st.columns(4)
            cols[0].metric("Stock Value", f"${final_stock_value:.2f}")
            cols[1].metric("Put Value", f"${final_put_value:.2f}")
            cols[2].metric("Total Value", f"${total_value:.2f}")
            cols[3].metric("Put Contracts", f"{num_puts:.2f}")
            
        except Exception as e:
            st.error("An error occurred:")
            st.exception(e)

if __name__ == "__main__":
    main()