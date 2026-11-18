import os
import pytest
pytestmark = pytest.mark.integration

@pytest.mark.parametrize("ticker", ["0005.HK", "0700.HK", "1299.HK"])
def test_alpha_vantage_news(ticker):
    import requests
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        pytest.skip("ALPHA_VANTAGE_API_KEY is required")
    response = requests.get("https://www.alphavantage.co/query", params={"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": key, "limit": 5}, timeout=15)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
