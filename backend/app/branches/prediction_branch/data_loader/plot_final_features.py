from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from sqlalchemy import func, and_
from app.core.logger import logger
from app.core.config import SessionLocal
from app.db.models import (
    FeaturesFinalPreparedV2, Company
)
import matplotlib.pyplot as plt

def plot_final_features_v1(company_id: int, save_path: str):
    db = SessionLocal()
    try:
        sql = "SELECT * FROM features_final_prepared WHERE company_id = %(cid)s ORDER BY date"
        df = pd.read_sql_query(sql, db.bind, params={"cid": company_id}, parse_dates=["date"])
        if df.empty:
            logger.warning(f"No data found for company_id {company_id} in features_final_prepared")
            return

        df.set_index("date", inplace=True)

        groups = [
            [
                ('close', 'Close Price', 'blue'),
                ('revenue', 'Revenue', 'orange'),
                ('operating_profit', 'Operating Profit', 'green'),
                ('gross_profit', 'Gross Profit', 'red'),
                ('net_profit', 'Net Profit', 'purple'),
            ],
            [
                ('ebitda', 'EBITDA', 'brown'),
                ('gdp', 'GDP', 'pink'),
                ('cpi', 'CPI', 'gray'),
                ('unemployment_rate', 'Unemployment Rate', 'olive'),
                ('interest_rate', 'Interest Rate', 'cyan'),
            ],
            [
                ('exchange_rate_eur', 'Exchange Rate EUR', 'magenta'),
                ('exchange_rate_usd', 'Exchange Rate USD', 'yellow'),
                ('sma_14', 'SMA 14', 'black'),
                ('ema_14', 'EMA 14', 'gray'),
                ('rsi_14', 'RSI 14', 'purple'),
            ],
            [
                ('macd', 'MACD', 'brown'),
                ('macd_signal', 'MACD Signal', 'pink'),
                ('macd_hist', 'MACD Hist', 'orange'),
            ]
        ]

        fig, axes = plt.subplots(len(groups), 1, figsize=(15, 5*len(groups)), sharex=True)

        if len(groups) == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            for col, label, color in group:
                if col in df.columns:
                    ax.plot(df.index, df[col], label=label, color=color)
                else:
                    logger.warning(f"Column '{col}' not found for company_id {company_id}")
            ax.set_ylabel("Value")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True)

        axes[0].set_title(f"Final Features for Company ID {company_id}")
        axes[-1].set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting features for company_id {company_id}: {e}")
    finally:
        db.close()

def plot_final_features_v2(company_id: int, save_path: str):
    db = SessionLocal()
    try:
        sql = "SELECT * FROM features_final_prepared_v2 WHERE company_id = %(cid)s ORDER BY date"
        df = pd.read_sql_query(sql, db.bind, params={"cid": company_id}, parse_dates=["date"])
        if df.empty:
            logger.warning(f"No data found for company_id {company_id} in features_final_prepared_v2")
            return

        df.set_index("date", inplace=True)

        groups = [
            [
                ('close', 'Close Price', 'blue'),
                ('revenue', 'Revenue', 'orange'),
                ('operating_profit', 'Operating Profit', 'green'),
                ('gross_profit', 'Gross Profit', 'red'),
                ('net_profit', 'Net Profit', 'purple'),
            ],
            [
                ('ebitda', 'EBITDA', 'brown'),
                ('gdp', 'GDP', 'pink'),
                ('cpi', 'CPI', 'gray'),
                ('unemployment_rate', 'Unemployment Rate', 'olive'),
                ('interest_rate', 'Interest Rate', 'cyan'),
            ],
            [
                ('exchange_rate_eur', 'Exchange Rate EUR', 'magenta'),
                ('exchange_rate_usd', 'Exchange Rate USD', 'yellow'),
            ]
        ]

        fig, axes = plt.subplots(len(groups), 1, figsize=(15, 5*len(groups)), sharex=True)

        if len(groups) == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            for col, label, color in group:
                if col in df.columns:
                    ax.plot(df.index, df[col], label=label, color=color)
                else:
                    logger.warning(f"Column '{col}' not found for company_id {company_id}")
            ax.set_ylabel("Value")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True)

        axes[0].set_title(f"Final Features for Company ID {company_id}")
        axes[-1].set_xlabel("Date")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting features for company_id {company_id}: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    plot_final_features_v1(company_id=550, save_path="company_features_v1.png")
    plot_final_features_v2(company_id=550, save_path="company_features_v2.png")
