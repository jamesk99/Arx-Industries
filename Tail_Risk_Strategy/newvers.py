import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np
import os
import plotly.graph_objects as go
from data import fetch_and_save_data


# -----------------------
# Custom Dark Theme CSS
# -----------------------
# Feel free to modify the colors, fonts, etc.
DARK_THEME_CSS = """
<style>
/* Main background and text */
body, .main {
    background-color: #222; 
    color: #fafafa;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}

/* Remove default Streamlit padding */
.block-container {
    padding: 1rem 2rem;
}

/* Hide default menu & header if desired */
header, footer, .viewerBadge_container__1QSob {
    visibility: hidden;
    height: 0;
    position: relative;
}

/* Sidebar styling */
.css-1aumxhk, .css-18e3th9, .css-1y4p8pa {
    background-color: #333 !important;
    color: #fafafa !important;
}

/* Sidebar icons / text style */
.css-qrbaxs, .css-1yk7n81 {
    color: #fafafa !important;
}

/* Tabs styling */
.stTabs [role="tablist"] button {
    background-color: #444;
    color: #fff;
    border: none;
}
.stTabs [role="tablist"] button[aria-selected="true"] {
    background-color: #666;
    color: #fff;
    border-bottom: 2px solid #0f0;
}
</style>
"""
def format_number(x):
    """Format numbers to be more readable"""
    if isinstance(x, float):
        if abs(x) > 1:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return x

def run_dashboard_app():
    # Configure page
    st.set_page_config(layout="wide", page_title="Portfolio Analytics Dashboard")
    # Inject custom CSS
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])

    with st.sidebar:
        st.subheader("Parameters")
        ticker = st.text_input("Ticker Symbol:", value="SPY", max_chars=5).upper()
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-08"))
        
        stock_allocation = st.text_input("Equity Allocation (%)", value=90.0)
        stock_allocation = float(stock_allocation)
        put_allocation = 100.0 - stock_allocation
        strike_pct = st.slider("Strike % OTM", min_value=0.0, max_value=100.0, value=80.0, help= "80 means strike is 20% below current price")
        time_to_expiry = st.text_input("Option Expiry (months)", value=6.0,)
        time_to_expiry = float(time_to_expiry)
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.25) / 100.0
        vol_override = st.slider("Vol Override (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100.0
        initial_price = st.text_input("Initial Price", value=400.00, max_chars=7)
        initial_price = float(initial_price)    
        st.image("https://canvasjs.com/wp-content/uploads/images/gallery/javascript-charts/overview/javascript-charts-graphs-index-data-label.png", caption="Invest")
        st.markdown("---")  # Just a divider
        # You can add more sections or placeholders as needed

    tabs = st.tabs(["Stocks", "Dividends"])

    # Let's populate the "Stocks" tab with the main content
    with tabs[0]:
        st.subheader("Stocks Overview")
        
        # TOP METRICS
        # Usually you'd calculate these from your own logic, 
        # e.g. total capital, total market value, total profit/loss, etc.
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Cost", value="$40,908.47")
        col2.metric(label="Market Value", value="$47,865.32")
        col3.metric(label="P/L", value="$6,956.85", delta="17.01%")

        # MIDDLE ROW: Cards for each "Stock N" 
        # (in reality, you’d loop through your list of holdings)
        st.write(" ")
        stock_cols = st.columns(5)
        stock_cols[0].metric("STOCK 1", "$19,677.24", "12.50%")
        stock_cols[1].metric("STOCK 2", "$8,008.96", "35.27%")
        stock_cols[2].metric("STOCK 3", "$7,494.55", "6.84%")
        stock_cols[3].metric("STOCK 4", "$8,201.82", "49.98%")
        stock_cols[4].metric("STOCK 5", "$4,482.75", "-10.56%")

        # BOTTOM ROW: Main content area
        fetch_and_save_data(ticker)
        
        data = pd.read_csv(f"{ticker}_stock_data.csv", index_col=0, parse_dates=True)
        pd.append(data)

        # Load the data from the CSV
        stock_data = pd.read_csv(f"{ticker}_stock_data.csv")
                
        # Format the display data
        display_df = stock_data.copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(format_number)

        # Show data sample
        st.subheader("Market Data")
        st.dataframe(
            pd.concat([display_df.head(2), display_df.tail(2)]).style.format(format_number),
            use_container_width=True,
            hide_index=True
        )
        
        # Calculate returns and volatility
        data['returns'] = data['Close'].pct_change(periods=7)
        data = data.dropna()
            
        current_price = float(data['Close'].iloc[-1])

        fig_line = px.line(
            data_frame=data,
            x='Date',
            y='Close',
            template="plotly_dark"
        )
        
        fig_line.update_layout(
            paper_bgcolor="#222",
            plot_bgcolor="#222", 
            font=dict(color="#FFF")
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # PIE CHARTS (Market Allocation, Sector Allocation)
        col_left, col_right = st.columns(2)

        # Market Allocation
        alloc_data = pd.DataFrame({
            "Region": ["USA", "Europe", "Asia", "Others"],
            "Allocation": [55, 20, 15, 10]
        })
        fig_alloc = px.pie(
            alloc_data, 
            names="Region", 
            values="Allocation", 
            hole=0.4, 
            title="Market Allocation",
            template="plotly_dark"
        )
        fig_alloc.update_layout(
            paper_bgcolor="#222", plot_bgcolor="#222",
            font=dict(color="#FFF")
        )
        col_left.plotly_chart(fig_alloc, use_container_width=True)

        # Sector Allocation
        sector_data = pd.DataFrame({
            "Sector": ["Tech", "Consumer", "Health", "Finance", "Energy"],
            "Percent": [35, 25, 15, 15, 10]
        })
        fig_sector = px.pie(
            sector_data, 
            names="Sector", 
            values="Percent", 
            hole=0.4, 
            title="Sector Allocation",
            template="plotly_dark"
        )
        fig_sector.update_layout(
            paper_bgcolor="#222", plot_bgcolor="#222",
            font=dict(color="#FFF")
        )
        col_right.plotly_chart(fig_sector, use_container_width=True)

    # Dividends tab can hold relevant info
    with tabs[1]:
        st.subheader("Dividends Overview")
        st.write("Some table or chart that tracks your dividends over time...")
        # CHART AREA: line chart or area chart for portfolio "Investment Value"
        # Just a dummy dataset for illustration
        chart2_data = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=12, freq="M"),
            "Value": [4000,3200,4100,6600,5700,5900,7000,7100,7200,8000,
                      6500,8000,]
        })
        fig2_line = px.line(chart2_data, x="Date", y="Value", 
                           title="Investment Value", template="plotly_dark")
        fig2_line.update_layout(
            paper_bgcolor="#222", plot_bgcolor="#222", 
            font=dict(color="#FFF")
        )
        st.plotly_chart(fig2_line, use_container_width=True)

        # PIE CHARTS (Market Allocation, Sector Allocation)
        col_left, col_right = st.columns(2)
        # Placeholder for user’s dividend data, calculations, etc.

# -------------------------------
# Run the entire app
# -------------------------------
if __name__ == "__main__":
    run_dashboard_app()
