from multiprocessing import Pool, cpu_count
from datetime import date

from boruta import BorutaPy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from sqlalchemy import select
from app.core.config import SessionLocal
from app.db.models import SelectedFeatures, Company, FeaturesFinalPrepared

import pandas as pd
import numpy as np
from tqdm import tqdm


def run_boruta_for_sector(sector_id: int):
    db = SessionLocal()

    try:
        # 1. Get company_ids
        companies = db.query(Company.id).filter(Company.sector_id == sector_id).all()
        company_ids = [company.id for company in companies]
        if not company_ids:
            return f"Sector {sector_id} — no companies"

        # 2. Query feature data directly using ORM
        query = db.query(FeaturesFinalPrepared).filter(
            FeaturesFinalPrepared.company_id.in_(company_ids),
            FeaturesFinalPrepared.date <= date(2024, 12, 31)
        )

        df = pd.read_sql(query.statement, db.bind)

        if df.empty:
            return f"Sector {sector_id} — no data"

        df = df.dropna(subset=["close"])
        if df.empty:
            return f"Sector {sector_id} — no target values (close)"

        # 3. Prepare X and y
        y = df["close"]
        X = df.drop(columns=["company_id", "date", "close"])
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.mean(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X)

        # 4. Fit Boruta
        forest = ExtraTreesRegressor(n_jobs=-1, n_estimators=200, max_depth=7, random_state=42)
        boruta = BorutaPy(estimator=forest, n_estimators='auto', max_iter=100, verbose=0, random_state=42)
        boruta.fit(X_scaled, y)

        selected = X.columns[boruta.support_].tolist()
        importances = boruta.ranking_.tolist()

        # 5. Save results
        record = SelectedFeatures(
            sector_id=sector_id,
            run_date=date.today(),
            model_type="boruta",
            selected_features=selected,
            total_features=len(X.columns),
            feature_importances=importances,
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
    db.close()

    if parallel:
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            for result in tqdm(pool.imap_unordered(run_boruta_for_sector, sectors), total=len(sectors)):
                print(result)
    else:
        for sector_id in tqdm(sectors):
            print(run_boruta_for_sector(sector_id))


if __name__ == "__main__":
    run_boruta_for_all_sectors(parallel=True)
