import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- CONFIG ----------------
BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
MAX_PAGES = 60
OUTPUT_DIR = "data"
OUTPUT_FILE = "shl_catalog.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; SHL-AI-Intern)"
}
# ----------------------------------------


def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,          # exponential backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


def get_soup(session, params):
    try:
        response = session.get(
            CATALOG_URL,
            params=params,
            timeout=40
        )
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"⚠️ Page failed for start={params.get('start')} → {e}")
        return None


def scrape_individual_tests():
    session = create_session()
    records = []
    start = 0

    for _ in range(MAX_PAGES):
        print(f"Scraping Individual Tests | start={start}")

        soup = get_soup(session, {
            "start": start,
            "type": 1   # ONLY Individual Test Solutions
        })

        if soup is None:
            print("⏳ Sleeping extra due to timeout...")
            time.sleep(10)
            start += PAGE_SIZE
            continue

        tables = soup.find_all("table")
        if not tables:
            print("❌ No table found, stopping.")
            break

        table = tables[0]
        rows = table.find_all("tr")[1:]

        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            link = cols[0].find("a")
            if not link:
                continue

            name = link.text.strip()
            url = urljoin(BASE_URL, link["href"])

            remote = "Yes" if cols[1].find("span", class_="green") else "No"
            adaptive = "Yes" if cols[2].find("span", class_="green") else "No"
            test_type = cols[3].text.strip()

            records.append({
                "assessment_name": name,
                "url": url,
                "remote_support": remote,
                "adaptive_support": adaptive,
                "test_type": test_type
            })

        start += PAGE_SIZE
        time.sleep(2)   # ⛔ IMPORTANT: slow scraping

    df = pd.DataFrame(records)
    return df.drop_duplicates(subset=["url"])


if __name__ == "__main__":
    df = scrape_individual_tests()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n✅ SUCCESS: Saved {len(df)} Individual Test Solutions")
    print(f"📁 File: {output_path}")
