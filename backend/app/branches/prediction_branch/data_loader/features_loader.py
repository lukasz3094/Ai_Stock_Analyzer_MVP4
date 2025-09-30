from typing import Iterable, List, Tuple
import pandas as pd
import numpy as np
from app.core.logger import logger
from app.core.config import SessionLocal
from app.db.models import Company, FeaturesFinalPrepared, FeaturesFinalPreparedV2, FeaturesFinalPreparedV3, SelectedFeatures, FeaturesPreparedWig20
from sqlalchemy import desc

ALWAYS_KEEP: Tuple[str, ...] = ("date", "company_id", "close", "sector_id")

# V1 with calculations

def _extract_feature_names(payload) -> List[str]:
    if payload is None:
        return []

    if isinstance(payload, list) and all(isinstance(x, str) for x in payload):
        return payload

    if isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
        return [k for k, _ in sorted(payload.items(), key=lambda kv: kv[1], reverse=True)]

    if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        keys_pref = ("feature", "name", "col", "column")
        out = []
        for d in payload:
            for k in keys_pref:
                if k in d and isinstance(d[k], str):
                    out.append(d[k])
                    break
        return out

    return []

def load_selected_feature_names_for_sector(sector_id: int, model_type: str = "boruta") -> List[str]:
    db = SessionLocal()
    try:
        sf = (
            db.query(SelectedFeatures)
              .filter(SelectedFeatures.sector_id == sector_id)
              .filter(SelectedFeatures.model_type == model_type)
              .order_by(desc(SelectedFeatures.run_date), desc(SelectedFeatures.id))
              .first()
        )
        if not sf:
            logger.warning(f"No selected features found for sector {sector_id}")
            return []
        names = _extract_feature_names(sf.selected_features)
        if not names:
            logger.warning(f"Selected features for sector {sector_id} exist but are empty/unrecognized format.")
        return names
    except Exception as e:
        logger.error(f"Error loading selected features for sector {sector_id}: {e}")
        return []
    finally:
        db.close()

def load_all_final_prepared_features_for_sector_v1(sector_id: int) -> pd.DataFrame:
    db = SessionLocal()
    try:
        companies = db.query(Company.id).filter(Company.sector_id == sector_id).all()
        company_ids = [cid for (cid,) in companies]
        if not company_ids:
            logger.warning(f"No companies found for sector {sector_id}")
            return pd.DataFrame()

        q = db.query(FeaturesFinalPrepared).filter(FeaturesFinalPrepared.company_id.in_(company_ids))
        df = pd.read_sql(q.statement, db.bind)

        if df.empty:
            logger.warning(f"No final prepared features found for sector {sector_id}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error loading final prepared features for sector {sector_id}: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def filter_final_by_selected(df: pd.DataFrame, selected: Iterable[str], always_keep: Tuple[str, ...] = ALWAYS_KEEP) -> pd.DataFrame:
    selected_set = set(selected) if selected else set()
    keep = [c for c in df.columns if (c in selected_set) or (c in always_keep)]

    missing = sorted(selected_set - set(df.columns))
    if missing:
        logger.warning(f"Missing columns in final_prepared for selected features: {missing}")

    return df.loc[:, keep].copy()

def get_data_for_model_per_sector_v1(sector_id: int, model_type: str = "boruta") -> pd.DataFrame:
    selected_names = load_selected_feature_names_for_sector(sector_id, model_type=model_type)
    final_df = load_all_final_prepared_features_for_sector_v1(sector_id)

    if final_df.empty:
        logger.warning(f"No data found for sector {sector_id}")
        return pd.DataFrame()

    if not selected_names:
        logger.warning(f"No selected feature names for sector {sector_id}; returning ONLY ALWAYS_KEEP columns.")
        return filter_final_by_selected(final_df, [], ALWAYS_KEEP)

    return filter_final_by_selected(final_df, selected_names, ALWAYS_KEEP)

# V2 without calculations

def load_all_final_prepared_features_for_sector_v2(sector_id: int) -> pd.DataFrame:
    db = SessionLocal()
    try:
        companies = db.query(Company.id).filter(Company.sector_id == sector_id).all()
        company_ids = [cid for (cid,) in companies]
        if not company_ids:
            logger.warning(f"No companies found for sector {sector_id}")
            return pd.DataFrame()

        q = db.query(FeaturesFinalPreparedV2).filter(FeaturesFinalPreparedV2.company_id.in_(company_ids))
        df = pd.read_sql(q.statement, db.bind)

        if df.empty:
            logger.warning(f"No final prepared features found for sector {sector_id}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error loading final prepared features for sector {sector_id}: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_data_for_model_per_sector_v2(sector_id: int, model_type: str = "boruta") -> pd.DataFrame:
    final_df = load_all_final_prepared_features_for_sector_v2(sector_id)

    if final_df.empty:
        logger.warning(f"No data found for sector {sector_id}")
        return pd.DataFrame()

    return final_df

# V3 with transformations
def load_all_final_prepared_features_for_sector_v3(sector_id: int) -> pd.DataFrame:
    db = SessionLocal()
    try:
        company_ids = [cid for (cid,) in db.query(Company.id).filter(Company.sector_id == sector_id).all()]
        if not company_ids:
            logger.warning(f"No companies for sector {sector_id}")
            return pd.DataFrame()
        q = db.query(FeaturesFinalPreparedV3).filter(FeaturesFinalPreparedV3.company_id.in_(company_ids))
        df = pd.read_sql(q.statement, db.bind)
        if df.empty:
            logger.warning(f"No V3 features for sector {sector_id}")
        return df
    except Exception as e:
        logger.error(f"V3 load error sector {sector_id}: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_data_for_model_per_sector_v3(sector_id: int) -> pd.DataFrame:
    return load_all_final_prepared_features_for_sector_v3(sector_id)

# WIG20
def load_all_final_prepared_features_for_company(company_id: int) -> pd.DataFrame:
    db = SessionLocal()
    try:
        q = db.query(FeaturesPreparedWig20).filter(FeaturesPreparedWig20.company_id == company_id)
        df = pd.read_sql(q.statement, db.bind)

        if df.empty:
            logger.warning(f"No final prepared features found for company {company_id}")
            return pd.DataFrame()
        
        selectedFeatures = db.query(SelectedFeatures).filter(SelectedFeatures.company_id == company_id).first()
        if not selectedFeatures:
            logger.warning(f"No selected features found for company {company_id}")
            return df

        for col in df.columns:
            if col not in ALWAYS_KEEP and col not in (selectedFeatures.selected_features or []):
                df = df.drop(columns=[col])
        
        return df
    except Exception as e:
        logger.error(f"Error loading final prepared features for company {company_id}: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_data_for_model_per_company(company_id: int, model_type: str = "boruta") -> pd.DataFrame:
    final_df = load_all_final_prepared_features_for_company(company_id)
    if final_df.empty:
        logger.warning(f"No data found for company {company_id}")
        return pd.DataFrame()
    
    final_df = final_df.dropna(axis=1, how='all')

    if final_df.empty:
        logger.warning(f"No data found for company {company_id}")
        return pd.DataFrame()

    return final_df

