import y finance as yf

#Prompting user for the share name
STK = input("Enter share name: ")

#Fetching historical makret data
data=yf.Ticker(STK).history(period="1d")

#Extracting the last market price
last_market_price = data['Close'].iloc[-1]

#Displaying the last market price
print("Last market price:", last_market_price)

#Displaying the Stock
print("Stock:", STK)