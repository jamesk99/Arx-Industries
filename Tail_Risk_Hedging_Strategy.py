import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
from math import log, sqrt, exp
from scipy.stats import norm

st.set_page_config(layout="wide")  # Make the app full width

def format_number(x):
    """Format numbers to be more readable"""
    if isinstance(x, float):
        if abs(x) > 1:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return x

# -----------------------------------------
# 1. Black-Scholes function for put pricing
# -----------------------------------------
def black_scholes_put_price(S, K, T, r, sigma):
    """
    Calculate the theoretical price of a European put option using Black-Scholes.
    :param S: Current price of the underlying
    :param K: Strike price
    :param T: Time to maturity (in years)
    :param r: Risk-free interest rate (annualized)
    :param sigma: Volatility of underlying (annualized)
    :return: The Black-Scholes put option price
    """
    # handle case T=0 or sigma=0 gracefully
    if T == 0 or sigma == 0:
        return max(K - S, 0)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    # Cumulative distribution function for standard normal
    put_price = (K * exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return put_price

# -----------------------------------------
# 2. Streamlit App Begins
# -----------------------------------------
def main():
    st.title("Tail-Risk Hedging Strategy Dashboard")
    col1, col2 = st.columns([1, 3])

    # -----------------------------------------
    # 2a. User Inputs
    # -----------------------------------------
    st.sidebar.header("Strategy Parameters")
    ticker = st.sidebar.text_input("Underlying Ticker:", value="SPY")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-08"))
    
    # Allocation parameters
    stock_allocation = st.sidebar.slider("Percent Capital in Underlying Equity Assets (%)", min_value=0, max_value=100, value=90)
    put_allocation = 100 - stock_allocation
    
    # Option parameters
    strike_pct_out_of_money = st.sidebar.slider(
        "Strike % OTM (e.g. 80 means 20% below current price)?", 
        min_value=50, max_value=100, value=80
    )
    years_to_expiry = st.sidebar.slider(
        "Years to Option Expiry?", 
        min_value=0.1, max_value=3.0, value=1.0, step=0.1
    )
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)", 
        min_value=0.0, max_value=8.0, value=3.0, step=0.1
    ) / 100.0
    volatility_override = st.sidebar.slider(
        "Implied Volatility Override (%) [Optional]", 
        min_value=0.0, max_value=100.0, value=0.0, step=1.0
    ) / 100.0
    
    # More parameters can be added here for a more sophisticated model
    initial_price = st.sidebar.number_input(
        "Initial Price of the Underlying Asset ($)",
        min_value=0, max_value=999, value=450
    )

    try:

        # -----------------------------------------
        # 3. Data Retrieval
        # -----------------------------------------
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            st.error("No data found. Please try another ticker or date range.")
            return
        
        st.success(f"Data retrieved for {ticker}. Showing first few rows:")
        
        # Create a DataFrame with first 3 and last 3 rows
        first_three = data.head(3)
        last_three = data.tail(3)
        
        # Create a divider row with ellipsis
        divider = pd.DataFrame(
            {col: ['...'] for col in data.columns},
            index=['...']
        )
        
        # Concatenate the parts
        truncated_df = pd.concat([first_three, divider, last_three])

        st.dataframe(truncated_df, hide_index=False)
        
        # Calculate the price of the underlying Asset 
        current_price = data['Close'].iloc[-1]
        
        strike_price = current_price * (strike_pct_out_of_money / 100.0)
        
        # Daily returns
        data["returns"] = data["Close"].pct_change(1)
        data.dropna() # Changed to reassignment
        
        # Annualized volatility (rough approximation from daily returns)
        daily_vol = data["returns"].std()
        ann_vol_est = daily_vol * np.sqrt(252)  # approximate 252 trading days
        ann_vol = volatility_override if volatility_override > 0 else ann_vol_est
        
        # Debug prints
        print("Data after calculating returns and dropping NaNs:")
        print(data.head(2))
        print(data.tail(2))
        print("Daily Volatility:", daily_vol)
        print("Annualized Volatility Estimate:", ann_vol_est)
        print("Current Price:", current_price)
        print("Strike Price:", strike_price)

        # -----------------------------------------
        # 4. Calculate Cost of Hedging (Put Price)
        # -----------------------------------------
        put_price = black_scholes_put_price(
            S=current_price,
            K=strike_price,
            T=years_to_expiry,
            r=risk_free_rate,
            sigma=ann_vol
        )
        
        st.write(f"**Current Price:** {current_price:,.2f}")
        st.write(f"**Strike Price:** {strike_price:,.2f}")
        st.write(f"**Black-Scholes Put Price:** {put_price:,.2f} (per share)")
        st.write(f"**Annualized Vol (est):** {ann_vol_est:.2%}")

        # -----------------------------------------
        # 5. Portfolio Simulation / Backtest
        #    Simplified Approach:
        #      - We'll do a "Buy and Hold" backtest for the stock portion
        #      - We won't dynamically hedge/roll in this example.
        #      - Show how the portfolio value might look if a tail event happened at the end.
        # -----------------------------------------
        #   NOTE: For a real approach, you'd want to do a rolling or scenario-based approach 
        #         but we'll keep it simpler here for demonstration.
        
        # Basic growth of stock portion from start_date to end_date
        data["stock_multiplier"] = (1 + data["returns"]).cumprod()
        
        # Assume we hold 1 share (to keep it simple) at the start
        data["stock_value"] = data["stock_multiplier"] * initial_price
        
        # For demonstration, let's suppose we buy puts at the start (one-time) 
        # and hold them through expiry. We'll see the hypothetical payoff at the end.
        
        # Number of shares to hedge: 
        #   If you have $100 total capital, you put 'stock_allocation' in SPY, so 
        #   your "notional" in SPY is $stock_allocation. Then you hedge that notional.
        #   In a real scenario, you'd decide how many shares you hold, etc.
        
        # The put payoff at maturity is max(K - S_T, 0) for each put. 
        # We'll simulate a scenario where a tail event happens near end_date.
        
        final_price = data["Close"].iloc[-1]
        put_payoff = max(strike_price - final_price, 0)
        
        # We'll imagine we purchase "X" puts such that we spend 'put_allocation' of capital on them.
        # For simplicity, let's say the total capital is $1. Then:
        total_capital = 1.0
        capital_in_stock = stock_allocation / 100.0
        capital_in_puts = put_allocation / 100.0
        
        # If put_price is 0 (theoretically not possible, but can be very small), handle it
        if put_price < 0.01:
            put_price = 0.01  # just a small floor for demonstration
        
        # Number of puts we can buy with capital_in_puts:
        num_puts = (capital_in_puts * total_capital) / put_price
        
        # Stock units purchased with capital_in_stock:
        # If we assume the "underlying" is also $1 total capital, let's assume 1 share for easy demonstration.
        # In reality, you’d scale to real dollar amounts or to the user’s actual portfolio size.
        #
        # For simplicity, let's do: user invests X dollars in stock, at the initial price.
        # => shares_owned = capital_in_stock / (initial_price / (some scaling factor)).
        
        # But let's keep it simple: 1 share.
        shares_owned = 1
        
        # On the final day, what's the portfolio value?
        final_stock_value = shares_owned * final_price
        final_put_value = num_puts * put_payoff
        final_portfolio_value = final_stock_value * (capital_in_stock) + final_put_value
        
        st.write("---")
        st.subheader("Hypothetical Final-Day Tail Scenario")
        st.write(f"**Final Price of {ticker}:** {final_price:,.2f}")
        st.write(f"**Put Payoff per Contract:** {put_payoff:,.2f}")
        st.write(f"**Number of Puts:** {num_puts:,.2f}")
        st.write(f"**Final Portfolio Value** (at the end, hypothetical): {final_portfolio_value:,.2f}")
        
        # -----------------------------------------
        # 6. Visualization
        # -----------------------------------------
        # Plot the stock price history
        fig = px.line(
            data, 
            x=data.index, 
            y="Close", 
            title=f"{ticker} History"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the running "stock_value" for demonstration
        fig_value = px.line(
            data, 
            x=data.index, 
            y="stock_value", 
            title="Hypothetical Stock Investment Value (Buy & Hold)"
        )
        st.plotly_chart(fig_value, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main Execution Block
if __name__ == "__main__":
    main()
