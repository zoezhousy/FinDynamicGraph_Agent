import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm


BASE_URL = "https://www1.hkexnews.hk"
SEARCH_URL = "https://www1.hkexnews.hk/search/titlesearch.xhtml"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www1.hkexnews.hk/search/titlesearch.xhtml",
}


def clean_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]


def search_hkex_announcements(
    stock_id: str,
    from_date: str = "20240101",
    to_date: str = "20260515",
    lang: str = "EN",
    title_keyword: str = "",
):
    """
    Search HKEXnews listed company documents.

    Parameters
    ----------
    stock_id:
        HKEXnews internal stockId, not stock code.
        Example: Tencent 00700 uses stockId=7609 in HKEXnews search URL.

    from_date:
        Format: YYYYMMDD

    to_date:
        Format: YYYYMMDD

    lang:
        EN or ZH

    title_keyword:
        Optional keyword in announcement title, e.g. "results", "profit warning"
    """

    params = {
        "MB-Daterange": "0",
        "category": "0",
        "documentType": "",
        "from": from_date,
        "lang": lang,
        "market": "SEHK",
        "searchType": "0",
        "stockId": stock_id,
        "t1code": "",
        "t2Gcode": "",
        "t2code": "",
        "title": title_keyword,
        "to": to_date,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    records = []

    # HKEXnews 页面里公告通常以超链接形式出现，PDF 链接一般包含 /listedco/listconews/
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/listedco/listconews/" not in href.lower():
            continue

        title = a.get_text(" ", strip=True)
        pdf_url = urljoin(BASE_URL, href)

        # 向上找父级文本，用于提取 release time / stock code / category
        parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else ""
        nearby_text = parent_text

        # 有些页面结构较深，向上多取几层
        p = a
        for _ in range(4):
            p = p.find_parent()
            if p:
                nearby_text = p.get_text(" ", strip=True)

        release_time = None
        stock_code = None
        stock_name = None

        m_time = re.search(r"Release Time:\s*([0-9]{2}/[0-9]{2}/[0-9]{4}\s+[0-9]{2}:[0-9]{2})", nearby_text)
        if m_time:
            release_time = m_time.group(1)

        m_code = re.search(r"Stock Code:\s*([0-9]{5})", nearby_text)
        if m_code:
            stock_code = m_code.group(1)

        m_name = re.search(r"Stock Short Name:\s*([A-Z0-9 \-&]+)", nearby_text)
        if m_name:
            stock_name = m_name.group(1).strip()

        records.append({
            "release_time": release_time,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "title": title,
            "pdf_url": pdf_url,
        })

    # 去重
    df = pd.DataFrame(records).drop_duplicates(subset=["pdf_url"])

    return df


def download_pdfs(df: pd.DataFrame, output_dir: str = "hkex_announcements"):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = row.get("title") or "announcement"
        release_time = row.get("release_time") or "unknown_time"
        stock_code = row.get("stock_code") or "unknown_code"
        pdf_url = row["pdf_url"]

        filename = clean_filename(f"{stock_code}_{release_time}_{title}.pdf")
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            continue

        try:
            r = requests.get(pdf_url, headers=HEADERS, timeout=60)
            r.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(r.content)

            time.sleep(0.5)

        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")


if __name__ == "__main__":
    # Tencent example: HKEXnews stockId=7609
    df = search_hkex_announcements(
        stock_id="7609",
        from_date="20250101",
        to_date="20260515",
        lang="EN",
        title_keyword="",  # 可改成 "results", "profit warning", "dividend" 等
    )

    print(df.head())
    print("Total announcements:", len(df))

    df.to_csv("hkex_00700_announcements_metadata.csv", index=False, encoding="utf-8-sig")

    download_pdfs(df, output_dir="hkex_00700_pdfs")