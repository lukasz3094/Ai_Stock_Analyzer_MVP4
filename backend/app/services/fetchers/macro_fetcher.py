import requests
import pandas as pd
from io import BytesIO
from lxml import html
from datetime import datetime, date
import xml.etree.ElementTree as ET
from app.core.config import SessionLocal
from sqlalchemy import select
from app.db.models import MacroData
from app.core.logger import logger

class MacroFetcher:
    def get_cpi_data(self):
        url = "https://static.nbp.pl/dane/inflacja/bazowa.xlsx"
        response = requests.get(url)
        df = pd.read_excel(BytesIO(response.content), skiprows=3)

        result = []
        for _, row in df.iterrows():
            dt = row[0]
            if pd.notna(dt) and isinstance(dt, (pd.Timestamp, datetime)):
                result.append({
                    "date": dt.date(),
                    "year": dt.year,
                    "month": dt.month,
                    "cpi": row[1]
                })
        return result

    def get_interest_rates_data(self):
        url = "https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml"
        response = requests.get(url)
        response.encoding = 'utf-8'
        root = ET.fromstring(response.content)

        result = []
        for pozycje in root.findall(".//pozycje"):
            date_str = pozycje.attrib.get("obowiazuje_od")
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            for pos in pozycje.findall("pozycja"):
                rate_id = pos.attrib.get("id")
                val_str = pos.attrib.get("oprocentowanie", "").replace(",", ".")
                try:
                    value = float(val_str)
                except ValueError:
                    continue

                if rate_id:
                    result.append({
                        "date": d,
                        "type": rate_id,
                        "value": value
                    })

        return result

    def get_unemployment_rates(self):
        result = []
        for i in range(4):
            d = datetime.today().replace(day=1) - pd.DateOffset(months=i)
            url = f"https://stat.gov.pl/obszary-tematyczne/rynek-pracy/bezrobocie-rejestrowane/stopa-bezrobocia-rejestrowanego-w-latach-1990-{d.year},{d.month},1.html"
            try:
                page = requests.get(url)
                tree = html.fromstring(page.content)
                table = tree.xpath("//table")[0]
                break
            except:
                continue
        else:
            return []

        for row in table.xpath(".//tr")[1:]:
            cells = row.xpath(".//td|.//th")
            if len(cells) < 13:
                continue
            try:
                year = int(cells[0].text_content().strip())
            except:
                continue
            for m in range(1, 13):
                val = cells[m].text_content().strip().replace(",", ".")
                try:
                    result.append({
                        "date": date(year, m, 1),
                        "value": float(val)
                    })
                except:
                    continue

        return sorted(result, key=lambda x: x["date"])

    def get_exchange_rates(self, currency: str):
        start = datetime(2002, 1, 1)
        end = datetime.today()
        delta = pd.DateOffset(days=92)
        results = []

        while start <= end:
            chunk_end = min(start + delta, end)
            url = f"https://api.nbp.pl/api/exchangerates/rates/a/{currency}/{start.date()}/{chunk_end.date()}/?format=json"
            try:
                res = requests.get(url)
                data = res.json()
                for rate in data["rates"]:
                    results.append({
                        "currency": currency.upper(),
                        "date": datetime.strptime(rate["effectiveDate"], "%Y-%m-%d").date(),
                        "value": rate["mid"]
                    })
            except:
                pass
            start = chunk_end + pd.DateOffset(days=1)

        return sorted(results, key=lambda x: x["date"])

    def get_gdp_data(self):
        url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/namq_10_gdp?geo=PL&unit=CLV10_MEUR&na_item=B1GQ&format=JSON"
        response = requests.get(url)
        doc = response.json()

        time_index_map = doc["dimension"]["time"]["category"]["index"]
        index_to_time_str = {str(v): k for k, v in time_index_map.items()}

        values = doc["value"]
        result = []

        for idx_str, val in values.items():
            time_str = index_to_time_str.get(idx_str)
            if not time_str or not isinstance(val, (int, float)):
                continue

            try:
                year = int(time_str[:4])
                quarter = int(time_str[6:])
            except (ValueError, IndexError):
                continue

            result.append({
                "year": year,
                "quarter": quarter,
                "country": "PL",
                "value": val
            })

        return result

def update_macro_cpi():
    db = SessionLocal()
    fetcher = MacroFetcher()

    try:
        records = fetcher.get_cpi_data()
        logger.info(f"Znaleziono {len(records)} rekordów CPI do aktualizacji.")

        for row in records:
            existing = db.execute(select(MacroData).where(MacroData.date == row["date"])).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=row["date"], cpi=row["cpi"]))
            else:
                existing.cpi = row["cpi"]
        db.commit()
        logger.info(f"Zaktualizowano dane CPI: {len(records)} rekordów.")
    except Exception as e:
        db.rollback()
        logger.error(f"Błąd podczas aktualizacji danych CPI: {e}")
    finally:
        db.close()

def update_macro_interest_rates():
    db = SessionLocal()
    fetcher = MacroFetcher()

    try:
        records = fetcher.get_interest_rates_data()
        logger.info(f"Znaleziono {len(records)} rekordów stóp procentowych do aktualizacji.")

        records = [row for row in records if row["type"] == "ref"]

        for row in records:
            existing = db.execute(select(MacroData).where(MacroData.date == row["date"])).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=row["date"], interest_rate=row["value"]))
            else:
                existing.interest_rate = row["value"]
        db.commit()
        logger.info(f"Zaktualizowano dane stóp procentowych: {len(records)} rekordów.")
    except Exception as e:
        db.rollback()
        logger.error(f"Błąd podczas aktualizacji danych stóp procentowych: {e}")
    finally:
        db.close()

def update_macro_unemployment_rates():
    db = SessionLocal()
    fetcher = MacroFetcher()

    try:
        records = fetcher.get_unemployment_rates()
        logger.info(f"Znaleziono {len(records)} rekordów stopy bezrobocia do aktualizacji.")

        for row in records:
            existing = db.execute(select(MacroData).where(MacroData.date == row["date"])).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=row["date"], unemployment_rate=row["value"]))
            else:
                existing.unemployment_rate = row["value"]
        db.commit()
        logger.info(f"Zaktualizowano dane stopy bezrobocia: {len(records)} rekordów.")
    except Exception as e:
        db.rollback()
        logger.error(f"Błąd podczas aktualizacji danych stopy bezrobocia: {e}")
    finally:
        db.close()

def update_macro_exchange_rates():
    db = SessionLocal()
    fetcher = MacroFetcher()

    try:
        eur_records = fetcher.get_exchange_rates("eur")
        usd_records = fetcher.get_exchange_rates("usd")
        logger.info(f"Znaleziono {len(eur_records)} rekordów kursów EUR i {len(usd_records)} rekordów kursów USD do aktualizacji.")

        pd.DataFrame(eur_records).to_csv("exchange_rates_eur.csv", index=False)
        pd.DataFrame(usd_records).to_csv("exchange_rates_usd.csv", index=False)

        for row in eur_records:
            existing = db.execute(select(MacroData).where(MacroData.date == row["date"])).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=row["date"], exchange_rate_eur=row["value"]))
            else:
                existing.exchange_rate_eur = row["value"]

        for row in usd_records:
            existing = db.execute(select(MacroData).where(MacroData.date == row["date"])).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=row["date"], exchange_rate_usd=row["value"]))
            else:
                existing.exchange_rate_usd = row["value"]

        db.commit()
        logger.info(f"Zaktualizowano dane kursów walut: {len(eur_records) + len(usd_records)} rekordów.")
    except Exception as e:
        db.rollback()
        logger.error(f"Błąd podczas aktualizacji danych kursów walut: {e}")
    finally:
        db.close()

def update_macro_gdp_data():
    db = SessionLocal()
    fetcher = MacroFetcher()

    try:
        records = fetcher.get_gdp_data()
        logger.info(f"Znaleziono {len(records)} rekordów PKB do aktualizacji.")

        for row in records:
            month = row["quarter"] * 3
            date_obj = date(row["year"], month, 1)
            existing = db.execute(select(MacroData).where(MacroData.date == date_obj)).scalar_one_or_none()
            if not existing:
                db.add(MacroData(date=date_obj, gdp=row["value"]))
            else:
                existing.gdp = row["value"]

        db.commit()
        logger.info(f"Zaktualizowano dane PKB: {len(records)} rekordów.")
    except Exception as e:
        db.rollback()
        logger.error(f"Błąd podczas aktualizacji danych PKB: {e}")
    finally:
        db.close()


def update_macro_data_table():
    
    update_macro_cpi()
    update_macro_interest_rates()
    update_macro_unemployment_rates()
    update_macro_exchange_rates()
    update_macro_gdp_data()

    logger.info(f"Zaktualizowano dane makroekonomiczne.")
