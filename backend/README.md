# Uruchomienie

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download pl_core_news_lg
uvicorn app.main:app --reload