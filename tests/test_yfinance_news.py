import yfinance as yf

for ticker in ['0005.HK', '0700.HK', '1299.HK']:
    stock = yf.Ticker(ticker)
    news = stock.get_news(count=5)
    print(f'\n{ticker}: {len(news) if news else 0} articles')
    if news:
        for i, a in enumerate(news[:3]):
            content = a.get('content', {})
            print(f'  [{i+1}] {content.get("provider",{}).get("displayName","?")} | {content.get("pubDate","")[:10]} | {content.get("title","")[:70]}')
