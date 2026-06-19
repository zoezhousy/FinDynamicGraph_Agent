import requests

API_KEY = "YOUR_ALPHA_VANTAGE_KEY"

url = "https://www.alphavantage.co/query"
params = {
    "function": "NEWS_SENTIMENT",
    "tickers": "NVDA",   
    "sort": "LATEST",
    "limit": 50,
    "apikey": API_KEY
}

r = requests.get(url, params=params)
data = r.json()
print(data)


# cannot recognize the ticker format in HK market
# can only get news for US stocks, e.g. AAPL, MSFT, etc. but not 0700.HK, 0005.HK, etc.
