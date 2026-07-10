import os
import requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("ALPHA_VANTAGE_API_KEY")
print(f"Key: {key[:4]}...{key[-4:]}")

for ticker in ['0005.HK', '0700.HK', '1299.HK']:
    r = requests.get('https://www.alphavantage.co/query', params={
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'apikey': key,
        'limit': 5,
    }, timeout=15)
    d = r.json()
    if 'feed' in d:
        print(f"\n✅ {ticker}: {len(d['feed'])} articles")
        for item in d['feed'][:3]:
            print(f"  {item.get('source','?')} | {item.get('time_published','')[:10]} | {item.get('title','')[:65]}")
            ts = item.get('ticker_sentiment', [])
            for t in ts:
                print(f"    → {t.get('ticker','?')}: {t.get('ticker_sentiment_label','?')} (score: {t.get('ticker_sentiment_score','?')})")
    elif 'Information' in d or 'Note' in d:
        print(f"\n⚠️ {ticker}: {d.get('Information', d.get('Note'))}")
    else:
        print(f"\n❌ {ticker}: {str(d)[:150]}")
