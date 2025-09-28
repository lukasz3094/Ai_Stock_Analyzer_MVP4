import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import unicodedata
from datetime import datetime
from app.core.config import SessionLocal
from app.db.models import Company, Fundamentals
from app.core.logger import logger
from app.services.getters.companies_getter import get_companies_by_group

def normalize(col):
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = re.sub(r"\(tys.\.\)\*?", "", col)
    col = col.replace("*", "").replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
    return col.lower()

def map_companies_fundamentals_columns(company, df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapuje surowe kolumny (po normalize) na jednolite pola:
    date, revenue, operating_profit, gross_profit, net_profit, ebitda
    + (opcjonalnie) assets, equity, shares, eps, bvps, audited

    Nie rzuca wyjątku, gdy kolumna nie istnieje – zostawia NaN.
    Dla ubezpieczycieli (sector_id == 22) revenue = składka + przychody z lokat (jeśli są).
    Dla banków (sector_id == 7) revenue = 'Wynik na działalności bankowej (tys.)'
      albo (fallback) suma: 'Przychody z tytułu odsetek (tys.)' + 'Przychody z tytułu prowizji (tys.)'.
    """
    def nrm(x: str) -> str:
        return normalize(x)

    # znormalizowana_nazwa -> oryginalna_nazwa
    norm2orig = {nrm(c): c for c in df.columns}

    def find_one(candidates: list[str], prefix_ok: bool = False) -> str | None:
        """Zwraca oryginalną nazwę pierwszej znalezionej kolumny (po normalize)."""
        for cand in candidates:
            if not prefix_ok and cand in norm2orig:
                return norm2orig[cand]
            if prefix_ok:
                for nc, orig in norm2orig.items():
                    if nc.startswith(cand):
                        return orig
        return None

    # kandydaci (po normalize)
    rev_candidates = [
        "przychodynettozesprzedazytys",
        "przychodyzesprzedazytys",
        "przychodytys",
        "przychodynetttys",
        "obrotytys",
        "przychodyogolemtys",
        # alternatywy sektorowe:
        "wynikdzialalnoscibankowejtys",   # banki
        "skladkanaudzialewlasnymtys",     # ubezpieczenia
        "skladkabruttotys",
    ]
    rev_insurance_parts = {
        "skladka": "skladkanaudzialewlasnymtys",
        "lokaty": "przychodyzlokattys",
    }
    # bank – składniki do fallbacku przy revenue
    bank_rev_parts = {
        "odsetki": ["przychodyztytuloodsetektys", "przychodyztytuoodsetektys"],
        "prowizje": ["przychodyztytuloprowizjitys"],
    }

    op_candidates = [
        "zyskstratazdziaalopertys",
        "zyskoperacyjnytys",
        "wynikzdzialalnoscioperacyjnejtys",
        "ebittys",  # czasem raportują EBIT zamiast "zysk oper."
        "wyniktechniczny",  # prefiks – ubezpieczenia
    ]
    gross_candidates = ["zyskstratabruttotys"]
    net_candidates   = ["zyskstratanettotys", "zyskstratanetto"]
    ebitda_candidates = ["ebitdatys", "ebitda"]
    amort_candidates  = ["amortyzacjatys", "amortyzacja"]

    assets_candidates = ["aktywatys", "aktywa"]
    equity_candidates = ["kapitalwlasnytys", "kapitalwlasnytys*", "kapitalwlasny"]
    shares_candidates = ["liczbaakcjitysszt", "liczbaakcjitysszt.", "liczbaakcji"]
    eps_candidates    = ["zysknaakcjez", "zysknaakcjezl", "zysknaakcje(zl)"]
    bvps_candidates   = ["wartoscksiegowanaakcjez", "wartoscksiegowanaakcjezl"]
    audited_candidates = ["raportzbadanyprzezaudytora"]

    out = pd.DataFrame(index=df.index).assign(date=df["date"])
    sector = getattr(company, "sector_id", None)

    # ---------- REVENUE ----------
    if sector == 22:  # ubezpieczenia
        sk_col = find_one([rev_insurance_parts["skladka"]])
        lk_col = find_one([rev_insurance_parts["lokaty"]])
        if sk_col or lk_col:
            rev = pd.Series(0.0, index=df.index, dtype="float64")
            if sk_col is not None:
                rev = rev.add(pd.to_numeric(df[sk_col], errors="coerce"), fill_value=0)
            if lk_col is not None:
                rev = rev.add(pd.to_numeric(df[lk_col], errors="coerce"), fill_value=0)
            out["revenue"] = rev
        else:
            col = find_one(rev_candidates)
            out["revenue"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    elif sector == 7:  # banki
        # priorytet: wynik na działalności bankowej
        col = find_one(["wyniknadzialalnoscibankowejtys", "wynikdzialalnoscibankowejtys",
                        "wyniknadzialalnoscbankowejtys"])
        if col:
            out["revenue"] = pd.to_numeric(df[col], errors="coerce")
        else:
            # fallback: odsetki + prowizje (jeśli dostępne)
            rev = None
            for cand in bank_rev_parts["odsetki"]:
                ocol = find_one([cand])
                if ocol:
                    rev = pd.to_numeric(df[ocol], errors="coerce")
                    break
            for cand in bank_rev_parts["prowizje"]:
                pcol = find_one([cand])
                if pcol:
                    rev = (rev.fillna(0) if rev is not None else 0) + pd.to_numeric(df[pcol], errors="coerce").fillna(0)
                    break
            out["revenue"] = rev if rev is not None else pd.NA
    else:  # spółki niefinansowe
        col = find_one(rev_candidates)
        out["revenue"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    # ---------- OPERATING PROFIT ----------
    if sector == 7:
        # Banki zwykle nie mają klasycznego EBIT – zostaw NaN (lub podmień na proxy, jeśli chcesz).
        out["operating_profit"] = pd.NA
    elif sector == 22:
        col = find_one(["wyniktechniczny"], prefix_ok=True)
        out["operating_profit"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA
    else:
        col = find_one(op_candidates, prefix_ok=True)
        out["operating_profit"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    # ---------- GROSS / NET ----------
    col = find_one(gross_candidates)
    out["gross_profit"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    col = find_one(net_candidates)
    out["net_profit"] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    # ---------- EBITDA ----------
    if sector in (7, 22):
        out["ebitda"] = pd.NA
    else:
        col = find_one(ebitda_candidates)
        if col:
            out["ebitda"] = pd.to_numeric(df[col], errors="coerce")
        else:
            ebit_col  = find_one(["ebittys", "zyskoperacyjnytys", "zyskstratazdziaalopertys"])
            amort_col = find_one(amort_candidates)
            if ebit_col and amort_col:
                out["ebitda"] = (
                    pd.to_numeric(df[ebit_col], errors="coerce").astype(float)
                    + pd.to_numeric(df[amort_col], errors="coerce").astype(float)
                )
            else:
                out["ebitda"] = pd.NA

    col = find_one(assets_candidates)
    if col: out["assets"] = pd.to_numeric(df[col], errors="coerce")
    col = find_one(equity_candidates)
    if col: out["equity"] = pd.to_numeric(df[col], errors="coerce")
    col = find_one(shares_candidates)
    if col: out["shares"] = pd.to_numeric(df[col], errors="coerce")
    col = find_one(eps_candidates)
    if col: out["eps"] = pd.to_numeric(df[col], errors="coerce")
    col = find_one(bvps_candidates)
    if col: out["bvps"] = pd.to_numeric(df[col], errors="coerce")
    col = find_one(audited_candidates, prefix_ok=True)
    if col: out["audited"] = df[col]

    needed = ["revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"]
    missing = [k for k in needed if k not in out.columns or out[k].isna().all()]
    if missing:
        logger.info(f"[{getattr(company, 'ticker_bankier', '?')}]: brak/metadane dla {missing} – zapisze NULL w DB.")

    return out

def fetch_fundamentals_from_bankier(ticker: str) -> pd.DataFrame:
    kwartal_map = {"I": "03-31", "II": "06-30", "III": "09-30", "IV": "12-31"}

    # --- DB: spółka + latest_date jako Timestamp ---
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

    # --- scraping ---
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

        # --- nagłówki ---
        headers_row = table.find("thead").find_all("th")
        headers_text = [h.get_text(strip=True) for h in headers_row]
        first = (headers_text[0] or "").strip().lower()
        if first not in {"kwartał", "kwartal", "okres", "okres sprawozdawczy"}:
            headers_text[0] = "kwartal"

        # --- wiersze ---
        rows = table.find("tbody").find_all("tr")
        data = []
        for row in rows:
            cols = row.find_all(["td", "th"])
            # uwaga: nie usuwamy znaku '-' (może oznaczać wartości ujemne)
            row_data = [
                col.get_text(strip=True)
                   .replace('\xa0', '')
                   .replace(' ', '')
                   .replace(',', '.')
                for col in cols
            ]
            data.append(row_data)

        # --- DataFrame + transpozycja ---
        df = pd.DataFrame(data)
        df.columns = headers_text
        df.set_index(df.columns[0], inplace=True)
        df = df.T.rename_axis('kwartal').reset_index()

        # --- budowa date z 'kwartal' ---
        kwartaly = df["kwartal"].str.extract(r'([IV]+)')
        lata = df["kwartal"].str.extract(r'(\d{4})')
        # "Q 0000" / "Q0000" odfiltruj
        mask = kwartaly[0].isin(kwartal_map.keys()) & lata[0].notna() & ~df["kwartal"].str.replace(" ", "").eq("Q0000")
        df = df[mask].copy()

        df["date"] = pd.to_datetime(
            lata[0] + "-" + kwartaly[0].map(kwartal_map),
            format="%Y-%m-%d",
            errors="coerce",
        )
        df = df[df["date"].notna()].copy()
        df.drop(columns=["kwartal"], inplace=True)

        # --- bez duplikatów w obrębie scrapu (nakładanie stron) ---
        new_mask = ~df["date"].isin(seen_dates)
        if not new_mask.all():
            logger.info(f"[{ticker}]: Pominięto {(~new_mask).sum()} duplikatów dat na stronie {page}.")
        df = df[new_mask]
        seen_dates.update(df["date"])

        if df.empty:
            page += 1
            continue

        # --- zatrzymaj, gdy strona nie ma nic nowszego ---
        if latest_date is not None and df["date"].max() <= latest_date:
            logger.info(f"[{ticker}]: No newer data on page {page}, stopping.")
            break

        all_dfs.append(df)
        page += 1

    if not all_dfs:
        return pd.DataFrame()

    # --- złączenie stron + drugi bezpiecznik na duplikaty ---
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # --- liczby: konwertuj kolumny poza 'date' na numery ---
    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- normalizacja nagłówków (zostaw 'date' bez zmian) ---
    df.columns = [c if c == "date" else normalize(c) for c in df.columns]

    # --- mapowanie na docelowe pola ---
    df = map_companies_fundamentals_columns(company, df)

    # --- ujednolicenie dat do północy ---
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # --- pomiń wiersze już obecne w DB (jeszcze jeden bezpiecznik) ---
    if not df.empty:
        session = SessionLocal()
        try:
            existing = (
                session.query(Fundamentals.date)
                .filter(
                    Fundamentals.company_id == company.id,
                    Fundamentals.date.in_(df["date"].unique().tolist()),
                )
                .all()
            )
            existing_dates = {d[0] for d in existing}
        finally:
            session.close()

        if existing_dates:
            before = len(df)
            df = df[~df["date"].isin(existing_dates)].copy()
            if before != len(df):
                logger.info(f"[{ticker}]: Pominieto {before - len(df)} istniejących rekordów (duplikaty w DB).")

    # --- dodatkowy filtr po latest_date (opcjonalny) ---
    if latest_date is not None:
        df = df[df["date"] > latest_date]

    # --- KONWERSJA braków: NaN/NA -> None (żeby psycopg2 nie zgłaszał NAType) ---
    if not df.empty:
        num_like = [c for c in ["revenue","operating_profit","gross_profit","net_profit","ebitda",
                                "assets","equity","shares","eps","bvps"]
                    if c in df.columns]
        for c in num_like:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # ustaw typ object i zamień NaN na None
        if num_like:
            df = df.astype({c: "object" for c in num_like})
            for c in num_like:
                df[c] = df[c].where(pd.notna(df[c]), None)

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

def update_fundamentals_selected_companies(company_ids):
    db = SessionLocal()
    companies = db.query(Company).filter(Company.id.in_(company_ids)).all()

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

if __name__ == "__main__":
    companies = get_companies_by_group("wig20")
    company_ids = [company.id for company in companies]
    update_fundamentals_selected_companies(company_ids)
