# Uruchomienie

python -m venv venv
.\venv\Scripts\activate
python -m app.branches.news_branch.pipeline
pip install -r requirements.txt
python -m spacy download pl_core_news_lg
uvicorn app.main:app --reload

mlflow ui --host 127.0.0.1 --port 5000

macos:
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn python-dotenv pydantic sqlalchemy psycopg2-binary mlflow prefect pandas==2.2.3 numpy==1.24.4 joblib tqdm lxml yfinance pandas_ta scikit-learn shap boruta transformers tokenizers lightgbm xgboost catboost torch torchvision torchaudio torch-geometric statsmodels