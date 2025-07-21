import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import unicodedata
from datetime import datetime
from app.core.config import SessionLocal
from app.db.models import Company, Fundamentals
from app.core.logger import logger

def normalize(col):
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = re.sub(r"\(tys.\.\)\*?", "", col)
    col = col.replace("*", "").replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
    return col.lower()

def map_companies_fundamentals_columns(company: Company, df: pd.DataFrame) -> pd.DataFrame:
    if company.sector_id == 7: # Banks
        df = df[[
            "date",
            "przychodyztytuuodsetektys",
            "wyniknadziaalnoscibankowejtys",
            "zyskstratabruttotys",
            "zyskstratanettotys",
            "amortyzacjatys"
        ]]
        df.columns = [
            "date",
            "revenue",
            "operating_profit",
            "gross_profit",
            "net_profit",
            "ebitda"
        ]
    else:
        df = df[[
            "date",
            "przychodynettozesprzedazytys",
            "zyskstratazdziaalopertys",
            "zyskstratabruttotys",
            "zyskstratanettotys",
            "ebitdatys"
        ]]
        df.columns = [
            "date",
            "revenue",
            "operating_profit",
            "gross_profit",
            "net_profit",
            "ebitda"
        ]

    return df


def fetch_fundamentals_from_bankier(ticker: str) -> pd.DataFrame:
    kwartal_map = {"I": "03-31", "II": "06-30", "III": "09-30", "IV": "12-31"}

    db = SessionLocal()
    company = db.query(Company).filter_by(ticker_bankier=ticker).first()
    if not company:
        db.close()
        raise ValueError(f"Spółka {ticker} nie znaleziona w bazie.")

    latest = (
        db.query(Fundamentals)
        .filter_by(company_id=company.id)
        .order_by(Fundamentals.date.desc())
        .first()
    )
    latest_date = latest.date if latest else None
    db.close()

    ticker = ticker.upper()
    base_url = f"https://www.bankier.pl/gielda/notowania/akcje/{ticker}/wyniki-finansowe/jednostkowy/kwartalny/standardowy/"
    headers = {"User-Agent": "Mozilla/5.0"}
    all_dfs = []
    page = 1

    while True:
        url = base_url + str(page)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Sprawdzenie, czy strona zawiera dane
        if "Nie znaleziono strony" in soup.text or "Brak wyników finansowych" in soup.text:
            logger.warning(f"[{ticker}]: Brak danych finansowych na stronie {url}.")
            return pd.DataFrame()

        box = soup.find("div", class_="boxContent boxTable")
        if not box:
            break

        table = box.find("table")
        if not table:
            break

        headers_row = table.find("thead").find_all("th")
        headers_text = [h.get_text(strip=True) for h in headers_row]

        rows = table.find("tbody").find_all("tr")
        data = []
        for row in rows:
            cols = row.find_all(["td", "th"])
            row_data = [col.get_text(strip=True).replace('\xa0', '').replace(' ', '').replace(',', '.').replace('-', '') for col in cols]
            data.append(row_data)

        df = pd.DataFrame(data)
        df.columns = headers_text
        df.set_index(df.columns[0], inplace=True)
        df = df.T.reset_index().rename(columns={"index": "kwartal"})

        kwartal_map = {"I": "03-31", "II": "06-30", "III": "09-30", "IV": "12-31"}
        kwartaly = df["kwartal"].str.extract(r'([IV]+)')
        lata = df["kwartal"].str.extract(r'(\d{4})')

        mask = kwartaly[0].isin(kwartal_map.keys()) & lata[0].notna()
        df = df[mask].copy()

        df["data"] = kwartaly[0].map(kwartal_map) + "-" + lata[0]
        df["date"] = pd.to_datetime(df["data"], format="%m-%d-%Y", errors="coerce")
        df = df[df["date"].notna()]
        df.drop(columns=["kwartal", "data"], inplace=True)

        if latest_date and (df["date"] <= latest_date).all():
            logger.info(f"[{ticker}]: No newer data on page {page}, stopping.")
            break

        all_dfs.append(df)

        page += 1

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Przekształcanie danych liczbowych
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce') * 1000

    # Normalizacja nazw kolumn
    df.columns = [normalize(c) for c in df.columns]

    # Przetwarzanie daty z kwartału
    kwartaly = df["kwartal"].str.extract(r'([IV]+)')
    lata = df["kwartal"].str.extract(r'(\d{4})')

    # Usuwanie wierszy z nieprawidłowym kwartałem lub rokiem
    mask = kwartaly[0].isin(kwartal_map.keys()) & lata[0].notna()
    df = df[mask].copy()

    df["data"] = kwartaly[0].map(kwartal_map) + "-" + lata[0]
    df["date"] = pd.to_datetime(df["data"], format="%m-%d-%Y", errors="coerce")

    # Usunięcie błędnych dat
    df = df[df["date"].notna()]

    df.drop(columns=["kwartal", "data"], inplace=True)

    # Wybór i zmiana nazw kolumn
    df = map_companies_fundamentals_columns(company, df)

    if latest_date:
        df = df[df["date"] > pd.Timestamp(latest_date)]

    return df

def insert_fundamentals(ticker: str):
    db = SessionLocal()
    company = db.query(Company).filter_by(ticker_bankier=ticker).first()
    if not company:
        logger.warning(f"Spółka {ticker} nie istnieje.")
        return

    df = fetch_fundamentals_from_bankier(ticker)

    for _, row in df.iterrows():
        record = Fundamentals(
            company_id=company.id,
            date=row["date"],
            revenue=row["revenue"],
            operating_profit=row["operating_profit"],
            gross_profit=row["gross_profit"],
            net_profit=row["net_profit"],
            ebitda=row["ebitda"]
        )
        db.merge(record)

    db.commit()
    db.close()
    logger.info(f"[{ticker}]: zapisano dane fundamentalne.")

def update_fundamentals_all_companies():
    db = SessionLocal()
    companies = db.query(Company).all()

    i = 1
    number_of_companies = len(companies)

    for company in companies:
        logger.info(f"({i} z {number_of_companies}) [{company.ticker_bankier}]: aktualizacja danych fundamentalnych...")

        i += 1

        try:
            insert_fundamentals(company.ticker_bankier)
        except Exception as e:
            logger.error(f"[{company.ticker_bankier}]: błąd podczas aktualizacji danych fundamentalnych: {e}")

    db.close()
    logger.info("Aktualizacja danych fundamentalnych zakończona.")
