import yfinance as yf

# Function to fetch and display stock details for multiple stocks
def fetch_multiple_stock_data(stock_symbols):
    for stock_name in stock_symbols:
        stock_name = stock_name.strip()  # Removing any leading/trailing whitespace
        try:
            # Fetching the ticker data
            ticker = yf.Ticker(stock_name)
            
            # Fetching historical market data for 1 day
            data = ticker.history(period="1d")
            
            # Checking if data exists
            if data.empty:
                print(f"\nNo data found for Stock: {stock_name}")
                continue
            
            # Extracting and displaying stock information
            stock_full_name = ticker.info.get("longName", "N/A")
            current_price = round(ticker.info.get("currentPrice", "N/A"), 2) if isinstance(ticker.info.get("currentPrice"), (int, float)) else "N/A"
            last_market_price = round(data['Close'].iloc[-1], 2)
            pe_ratio = round(ticker.info.get("trailingPE", "N/A"), 2) if isinstance(ticker.info.get("trailingPE"), (int, float)) else "N/A"
            price_book_ratio = round(ticker.info.get("priceToBook", "N/A"), 2) if isinstance(ticker.info.get("priceToBook"), (int, float)) else "N/A"
            market_cap = round(ticker.info.get("marketCap", "N/A"), 2) if isinstance(ticker.info.get("marketCap"), (int, float)) else "N/A"
            eps = round(ticker.info.get("trailingEps", "N/A"), 2) if isinstance(ticker.info.get("trailingEps"), (int, float)) else "N/A"
            profit_margin = round(ticker.info.get("profitMargins", "N/A"), 2) if isinstance(ticker.info.get("profitMargins"), (int, float)) else "N/A"
            dividend_yield = round(ticker.info.get("dividendYield", "N/A"), 2) if isinstance(ticker.info.get("dividendYield"), (int, float)) else "N/A"
            dividend_frequency = ticker.info.get("dividendFrequency", "N/A")

            # Displaying the information in a user-friendly way
            print(f"\n================= Stock Information for: {stock_name.upper()} ({stock_full_name}) =================")
            print(f"Current Price: {current_price}")
            print(f"Last Market Close Price: {last_market_price}")
            print(f"P/E Ratio: {pe_ratio}")
            print(f"Price/Book Ratio: {price_book_ratio}")
            print(f"Market Cap: {market_cap}")
            print(f"EPS: {eps}")
            print(f"Profit Margin: {profit_margin}")
            
            if dividend_yield != "N/A":
                print(f"Dividend Yield: {dividend_yield}")
                if dividend_frequency:
                    print(f"Dividend Frequency: {dividend_frequency}")
            print("================================================================================\n")
        
        except Exception as e:
            print(f"An error occurred for Stock: {stock_name}. Error: {e}")

# Prompting user for the stock names, separated by commas
stock_symbols = input("Enter stock symbols separated by commas (e.g., AAPL, TSLA, MSFT): ").split(',')

# Fetch and display the stock data for each symbol
fetch_multiple_stock_data(stock_symbols)