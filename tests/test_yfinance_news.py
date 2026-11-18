import pytest
pytestmark = pytest.mark.integration

@pytest.mark.parametrize("ticker", ["0005.HK", "0700.HK", "1299.HK"])
def test_yfinance_news(ticker):
    import yfinance as yf
    news = yf.Ticker(ticker).get_news(count=5)
    assert isinstance(news, list)
