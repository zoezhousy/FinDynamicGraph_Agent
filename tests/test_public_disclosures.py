from datetime import datetime

import pytest

from src.collectors.public_disclosures import parse_issuer_html
from src.eval.thesis_sources import LocalEvidence


def test_article_date_only_and_related_news_do_not_leak():
    html = '''<main><article><header><h1>Company partnership</h1>
    <div class="entry-date">September 5, 2024</div></header>
    <div class="entry-content">Tencent partnered with Example Company.
    This is an official historical announcement with enough body text to validate extraction.</div>
    </article><article>2026 New unrelated recommendations</article></main>'''
    record, audit = parse_issuer_html('0700.HK', 'https://www.tencent.com/example', html)
    assert '2026' not in record['text']
    assert audit['publication_date'] == '2024-09-05'
    evidence = LocalEvidence([record])
    assert not evidence.snapshot('0700.HK', datetime.fromisoformat('2024-09-05T16:10:00+08:00'))['evidences']
    assert evidence.snapshot('0700.HK', datetime.fromisoformat('2024-09-06T16:10:00+08:00'))['evidences']


@pytest.mark.parametrize('url', ['https://evil.example/story', 'http://www.tencent.com/story',
                                    'https://www.hsbc.com/story'])
def test_unknown_or_mismatched_issuer_rejected(url):
    with pytest.raises(ValueError, match='matching official'):
        parse_issuer_html('0700.HK', url, '<main>untrusted</main>')


def test_missing_article_date_cannot_use_recommendation_date():
    with pytest.raises(ValueError, match='publication date'):
        parse_issuer_html('0700.HK', 'https://www.tencent.com/story',
                          '<main><article><div class="entry-content">body</div></article>'
                          '<article><header><div class="entry-date">January 1, 2024</div></header></article></main>')
