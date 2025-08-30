from app.services.getters.companies_getter import get_all_company_aliases
from app.services.getters.sectors_getter import get_all_context_tags
import spacy
from spacy.matcher import PhraseMatcher
import torch
import gc

nlp = None
matcher = None
alias_map = {}
context_tags = []
sentiment_pipeline = None

model_name = "bardsai/twitter-sentiment-pl-base"

def init_worker():
    global nlp, matcher, alias_map, context_tags, sentiment_pipeline

    nlp = spacy.load("pl_core_news_lg", exclude=["parser", "tagger", "lemmatizer"])
    try:
        spacy.prefer_gpu()
    except:
        pass

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    company_aliases = [
        {"company_id": a.company_id, "alias": a.alias.lower()}
        for a in get_all_company_aliases()
    ]
    patterns = [nlp.make_doc(alias["alias"]) for alias in company_aliases]
    matcher.add("COMPANY_ALIASES", patterns)
    alias_map = {alias["alias"]: alias["company_id"] for alias in company_aliases}

    context_tags = [
        {"id": t.id, "name": t.name}
        for t in get_all_context_tags()
    ]

    device = 0 if torch.cuda.is_available() else -1

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

def classify_batch_in_process(news_batch):
    results = []
    for news_dict in news_batch:
        # Upewniamy się, że wszystkie pola są w formacie tekstowym
        headline = news_dict.get("headline", "")
        content = news_dict.get("content", "")
        text = f"{headline} {content[:100]}".lower()

        doc = nlp(text)

        # Sentiment analysis
        sentiment_result = sentiment_pipeline(text)[0]
        label = sentiment_result['label'].lower()
        score = sentiment_result['score']

        # Company alias matching
        matches = matcher(doc)
        company_ids = list({
            alias_map.get(str(doc[start:end]).lower())
            for _, start, end in matches
            if alias_map.get(str(doc[start:end]).lower()) is not None
        })

        # Context tag similarity
        tag_scores = {
            tag['id']: doc.similarity(nlp(tag['name'].lower()))
            for tag in context_tags
        }
        best_context_tag_id = max(tag_scores, key=tag_scores.get)
        confidence_score = tag_scores[best_context_tag_id]

        results.append({
            "news_article_id": news_dict["id"],
            "company_ids": list(set(company_ids)),
            "context_tag_id": best_context_tag_id,
            "confidence_score": confidence_score,
            "sentiment_score": score,
            "sentiment_label": label
        })

    gc.collect()
    return results

