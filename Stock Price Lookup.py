import yfinance as yf

# Prompting user for the share names, separated by commas
stock_symbols = input("Enter share names separated by commas: ").split(',')

# Iterating through each stock symbol
for STK in stock_symbols:
    STK = STK.strip()  # Removing any leading/trailing whitespace
    # Fetching historical market data
    data = yf.Ticker(STK).history(period="1d")
    # Extracting the last market price
    if not data.empty:
        last_market_price = data['Close'].iloc[-1]
        # Displaying the last market price
        print(f"Stock: {STK}, Last market price: {last_market_price}")
    else:
        print(f"No data found for Stock: {STK}")