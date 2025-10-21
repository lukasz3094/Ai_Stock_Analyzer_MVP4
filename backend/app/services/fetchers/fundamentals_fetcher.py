import requests
import pandas as pd
from app.services.getters.companies_getter import get_companies_by_group
from bs4 import BeautifulSoup
import re
import unicodedata
from datetime import datetime
from app.core.config import SessionLocal
from app.db.models import Company, Fundamentals
from app.core.logger import logger

# --- utils ---
def normalize(col: str) -> str:
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = re.sub(r"\(tys.\.\)\*?", "", col)
    col = (col.replace("*", "")
              .replace("\xa0", "")
              .replace(" ", "")
              .replace(".", "")
              .replace("(", "")
              .replace(")", ""))
    return col.lower()

def _find_first(norm2orig: dict[str, str], candidates: list[str], prefix_ok: bool = False) -> str | None:
    # exact
    for c in candidates:
        if c in norm2orig:
            return norm2orig[c]
    if prefix_ok:
        # prefix
        for c in candidates:
            for n, o in norm2orig.items():
                if n.startswith(c):
                    return o
    return None

def map_bank_columns(df: pd.DataFrame) -> pd.DataFrame:
    def nrm(x: str) -> str: return normalize(x)
    norm2orig = {nrm(c): c for c in df.columns}

    interest_candidates = [
        "przychodyztytuloodsetektys", "przychodyztytuoodsetektys",
        "przychodyztytuuodsetektys",
        "przychodyztytuloodsetek", "przychodyztytuuodsetek",
    ]
    fee_candidates = [
        "przychodyztytuloprowizjitys",
        "przychodyztytuuprowizjitys",
        "przychodyztytuloprowizji", "przychodyztytuuprowizji",
    ]
    bankres_candidates = [
        "wynikdzialalnoscibankowejtys",
        "wyniknadzialnoscibankowejtys",
        "wyniknadziaalnoscibankowejtys",
        "wyniknadzia", "wynikdzialalnoscibankowej",
    ]

    gross_candidates = ["zyskstratabruttotys", "zyskstratabrutto"]
    net_candidates   = ["zyskstratanettotys", "zyskstratanetto"]
    amort_candidates = ["amortyzacjatys", "amortyzacja"]

    assets_candidates = ["aktywatys", "aktywa"]
    equity_candidates = [
        "kapitalwlasnytys", "kapitalwlasny",
        "kapitawasnytys", "kapitawasny",
        "kapita", "kapitaw"
    ]
    shares_candidates = ["liczbaakcjitysszt", "liczbaakcjitysszt.", "liczbaakcji"]
    bvps_candidates   = ["wartoscksiegowanaakcjez", "wartoscksiegowanaakcjezl", "wartoscksiegowanaakcje"]
    eps_candidates    = ["zysknaakcjez", "zysknaakcjezl", "zysknaakcje"]

    out = pd.DataFrame(index=df.index).assign(date=df["date"])

    def take(cands):
        col = _find_first(norm2orig, cands, prefix_ok=True)
        return pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    out["interest_income"]  = take(interest_candidates)
    out["fee_income"]       = take(fee_candidates)
    out["banking_result"]   = take(bankres_candidates)
    out["gross_profit"]     = take(gross_candidates)
    out["net_profit"]       = take(net_candidates)
    out["amortization"]     = take(amort_candidates)
    out["assets"]           = take(assets_candidates)
    out["equity"]           = take(equity_candidates)
    out["shares_thousands"] = take(shares_candidates)
    out["bvps"]             = take(bvps_candidates)
    out["eps"]              = take(eps_candidates)

    return out

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
    latest_date = pd.Timestamp(latest.date) if latest else None
    db.close()

    ticker = ticker.upper()
    base_url = f"https://www.bankier.pl/gielda/notowania/akcje/{ticker}/wyniki-finansowe/jednostkowy/kwartalny/standardowy/"
    headers = {"User-Agent": "Mozilla/5.0"}

    all_dfs: list[pd.DataFrame] = []
    seen_dates: set = set()
    page = 1

    while True:
        url = base_url + str(page)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

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

        if not (headers_text and headers_text[0]):
            headers_text[0] = "pozycja"

        rows = table.find("tbody").find_all("tr")
        data = []
        for row in rows:
            cols = row.find_all(["td", "th"])
            row_data = [
                col.get_text(strip=True)
                   .replace('\xa0', '')
                   .replace(' ', '')
                   .replace(',', '.')
                for col in cols
            ]
            data.append(row_data)

        df = pd.DataFrame(data)
        df.columns = headers_text
        df.set_index(df.columns[0], inplace=True)
        df = df.T.rename_axis('kwartal').reset_index()

        # --- kwartał -> data ---
        kwartaly = df["kwartal"].str.extract(r'([IV]+)')
        lata = df["kwartal"].str.extract(r'(\d{4})')
        mask = kwartaly[0].isin(kwartal_map.keys()) & lata[0].notna() & ~df["kwartal"].str.replace(" ", "").eq("Q0000")
        df = df[mask].copy()
        df["date"] = pd.to_datetime(lata[0] + "-" + kwartaly[0].map(kwartal_map), format="%Y-%m-%d", errors="coerce")
        df = df[df["date"].notna()].copy()
        df.drop(columns=["kwartal"], inplace=True)

        new_mask = ~df["date"].isin(seen_dates)
        df = df[new_mask]
        seen_dates.update(df["date"])

        if df.empty:
            page += 1
            continue

        if latest_date is not None and df["date"].max() <= latest_date:
            logger.info(f"[{ticker}]: No newer data on page {page}, stopping.")
            break

        df.columns = [c if c == "date" else normalize(c) for c in df.columns]
        df = map_bank_columns(df)

        all_dfs.append(df)
        page += 1

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # ujednolicenie dat do północy
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if latest_date is not None:
        df = df[df["date"] > latest_date]

    # konwersja NaN -> None (dla Numeric)
    num_cols = [c for c in df.columns if c != "date"]
    if num_cols:
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.astype({c: "object" for c in num_cols})
        for c in num_cols:
            df[c] = df[c].where(pd.notna(df[c]), None)

    return df

def insert_fundamentals(ticker: str):
    db = SessionLocal()
    company = db.query(Company).filter_by(ticker_bankier=ticker).first()
    if not company:
        logger.warning(f"Spółka {ticker} nie istnieje.")
        db.close()
        return

    df = fetch_fundamentals_from_bankier(ticker)
    if df.empty:
        logger.info(f"[{ticker}]: brak nowych danych.")
        db.close()
        return

    for _, row in df.iterrows():
        record = Fundamentals(
            company_id=company.id,
            date=row["date"],
            interest_income=row.get("interest_income"),
            fee_income=row.get("fee_income"),
            banking_result=row.get("banking_result"),
            gross_profit=row.get("gross_profit"),
            net_profit=row.get("net_profit"),
            amortization=row.get("amortization"),
            assets=row.get("assets"),
            equity=row.get("equity"),
            shares_thousands=row.get("shares_thousands"),
            bvps=row.get("bvps"),
            eps=row.get("eps"),
        )
        db.merge(record)

    db.commit()
    db.close()
    logger.info(f"[{ticker}]: zapisano dane fundamentalne.")

if __name__ == "__main__":
    companies = get_companies_by_group("wig-banki")
    company_ids = [company.id for company in companies]

    for ticker in [company.ticker_bankier for company in companies]:
        insert_fundamentals(ticker)
