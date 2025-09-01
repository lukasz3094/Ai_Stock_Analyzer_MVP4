# Uruchomienie

python -m venv venv
.\venv\Scripts\activate
python -m app.branches.news_branch.pipeline
pip install -r requirements.txt
python -m spacy download pl_core_news_lg
uvicorn app.main:app --reload

mlflow ui --host 127.0.0.1 --port 5000