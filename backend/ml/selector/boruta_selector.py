from multiprocessing import Pool, cpu_count
from datetime import date

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sqlalchemy import select
from app.core.config import SessionLocal
from app.db.models import SelectedFeatures, Company, FeaturesPreparedWig20

from app.services.getters.companies_getter import get_companies_by_group

import pandas as pd
import numpy as np
from tqdm import tqdm


def run_boruta_for_sector(sector_id: int):
    db = SessionLocal()

    try:
        query = db.query(FeaturesPreparedWig20).filter(
            FeaturesPreparedWig20.sector_id == sector_id,
            FeaturesPreparedWig20.date <= date(2024, 12, 31)
        )

        df = pd.read_sql(query.statement, db.bind)

        if df.empty:
            return f"Sector {sector_id} — no data"

        df = df.dropna(subset=["close"])
        if df.empty:
            return f"Sector {sector_id} — no target values (close)"

        y = df["close"]
        X = df.drop(columns=["id", "company_id", "date", "close", "sector_id"])
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X)

        if X_scaled.shape[1] == 0:
            return f"Sector {sector_id} — no numeric features"

        forest = RandomForestRegressor(
            n_jobs=-1, n_estimators=200, max_depth=7, random_state=42
        )
        boruta = BorutaPy(
            estimator=forest,
            n_estimators='auto',
            max_iter=300,
            verbose=0,
            random_state=42
        )
        boruta.fit(X_scaled, y)

        selected_mask = boruta.support_
        if not any(selected_mask):
            return f"Sector {sector_id} — no features selected by Boruta"

        selected = X.columns[selected_mask].tolist()
        all_importances = {
            feature: int(rank)
            for feature, rank in zip(X.columns, boruta.ranking_)
        }

        record = SelectedFeatures(
            sector_id=sector_id,
            run_date=date.today(),
            model_type="boruta",
            selected_features=selected,
            total_features=len(X.columns),
            feature_importances=all_importances,
            notes=None
        )

        db.add(record)
        db.commit()

        return f"Sector {sector_id} — {len(selected)} features selected"

    except Exception as e:
        return f"Sector {sector_id} — ERROR: {str(e)}"
    finally:
        db.close()

def run_boruta_for_all_sectors(parallel=True):
    db = SessionLocal()
    stmt = select(Company.sector_id).distinct()
    sectors = db.execute(stmt).scalars().all()
    sectors = [int(s) for s in sectors if s is not None]
    db.close()

    if parallel:
        with Pool(processes=1) as pool:
            for result in tqdm(pool.imap_unordered(run_boruta_for_sector, sectors), total=len(sectors)):
                print(result)
    else:
        for sector_id in tqdm(sectors):
            print(run_boruta_for_sector(sector_id))

def run_boruta_for_company(company_id: int):
    if not isinstance(company_id, int):
        return

    db = SessionLocal()

    try:
        query = db.query(FeaturesPreparedWig20).filter(
            FeaturesPreparedWig20.company_id == company_id,
            FeaturesPreparedWig20.date <= date(2024, 12, 31)
        )

        df = pd.read_sql(query.statement, db.bind)
        df = df.drop(columns=["high", "low", "open", "volume"])

        if df.empty:
            return f"Company {company_id} — no data"

        sector_id = int(df["sector_id"].iloc[0])

        df = df.dropna(subset=["close"])
        if df.empty:
            return f"Company {company_id} — no target values (close)"

        y = df["close"]
        X = df.drop(columns=["id", "company_id", "date", "close", "sector_id"])
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X)

        if X_scaled.shape[1] == 0:
            return f"Company {company_id} — no numeric features"

        forest = RandomForestRegressor(
            n_jobs=-1, n_estimators=200, max_depth=7, random_state=42
        )
        boruta = BorutaPy(
            estimator=forest,
            n_estimators='auto',
            max_iter=300,
            verbose=0,
            random_state=42
        )
        boruta.fit(X_scaled, y)

        selected_mask = boruta.support_
        if not any(selected_mask):
            return f"Company {company_id} — no features selected by Boruta"

        selected = X.columns[selected_mask].tolist()
        all_importances = {
            feature: int(rank)
            for feature, rank in zip(X.columns, boruta.ranking_)
        }

        record = SelectedFeatures(
            company_id=int(company_id),
            sector_id=int(sector_id),
            run_date=date.today(),
            model_type="boruta",
            selected_features=selected,
            total_features=len(X.columns),
            feature_importances=all_importances,
            notes=None
        )

        db.add(record)
        db.commit()

        return f"Company {company_id} — {len(selected)} features selected"

    except Exception as e:
        return f"Company {company_id} — ERROR: {str(e)}"
    finally:
        db.close()

def run_boruta_for_wig_companies(parallel=True):
    companies = get_companies_by_group("wig20")
    company_ids = [company.id for company in companies]
    company_ids = [int(c) for c in company_ids if c is not None]

    if parallel:
        with Pool(processes=1) as pool:
            for result in tqdm(pool.imap_unordered(run_boruta_for_company, company_ids), total=len(company_ids)):
                print(result)
    else:
        for company_id in tqdm(company_ids):
            print(run_boruta_for_company(company_id))

if __name__ == "__main__":
    run_boruta_for_wig_companies(parallel=False)
