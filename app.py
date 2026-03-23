from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, abort, redirect, render_template, request, url_for

from newsapi_client import NewsApiClient, NewsApiError, NewsArticle
from model.predict import (
    build_tfidf_matrix,
    cluster_articles,
    filter_by_query,
    most_similar_articles,
)

# ✅ NEW: Import database functions
from database import create_table, save_articles, get_articles


load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

FALLBACK_ARTICLES: List[NewsArticle] = [
    NewsArticle(
        title="Demo: Cricket roundup and key takeaways",
        description="Sample article used when NewsAPI is unreachable.",
        content="This is fallback content so you can still demo search, clustering, and ML similarity.",
        url="",
        image_url="",
        source="Demo data",
        author="",
        published_at="",
    ),
    NewsArticle(
        title="Demo: AI in sports analytics is growing",
        description="Sample article about AI trends in sports analytics.",
        content="Teams increasingly apply machine learning to strategy, scouting, and performance analysis.",
        url="",
        image_url="",
        source="Demo data",
        author="",
        published_at="",
    ),
    NewsArticle(
        title="Demo: Technology updates this week",
        description="Sample technology digest for offline demo mode.",
        content="A summary of product launches, security patches, and developer tooling improvements.",
        url="",
        image_url="",
        source="Demo data",
        author="",
        published_at="",
    ),
    NewsArticle(
        title="Demo: Markets and business headlines",
        description="Sample business news card to test clustering.",
        content="Investors tracked macroeconomic indicators and sector performance across global markets.",
        url="",
        image_url="",
        source="Demo data",
        author="",
        published_at="",
    ),
]


COUNTRY_ALIASES = {
    "us": "us",
    "united states": "us",
    "in": "in",
    "india": "in",
    "gb": "gb",
    "united kingdom": "gb",
    "au": "au",
    "australia": "au",
    "ca": "ca",
    "canada": "ca",
}

COUNTRY_NAMES = {
    "us": "United States",
    "in": "India",
    "gb": "United Kingdom",
    "au": "Australia",
    "ca": "Canada",
}

ALLOWED_CATEGORIES = {
    "",
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology",
}


def _normalize_country(value: str) -> str:
    raw = (value or "").strip().lower()
    return COUNTRY_ALIASES.get(raw, "us")


def _normalize_category(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in ALLOWED_CATEGORIES:
        return raw
    return ""


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")

    # ✅ NEW: Initialize database
    create_table()

    api_key = os.environ.get("NEWSAPI_KEY", "")
    client: Optional[NewsApiClient]
    try:
        client = NewsApiClient(api_key=api_key)
    except ValueError:
        client = None

    def _get_articles() -> List[NewsArticle]:
        if client is None:
            raise NewsApiError("Missing NEWSAPI_KEY. Add it in a .env file.")

        mode = (request.args.get("mode") or "top").strip()
        q = (request.args.get("q") or "").strip()
        country = _normalize_country(request.args.get("country") or "us")
        category = _normalize_category(request.args.get("category") or "")

        if mode == "search":
            if not q:
                return []

            articles = client.everything(q=q, page_size=24)
            if not articles:
                articles = client.top_headlines(country=country, category=category, q=q, page_size=24)
            return articles

        # Top mode must use only country/category (keyword is ignored).
        q = ""
        articles = client.top_headlines(country=country, category=category, page_size=24)

        # Keep top mode country-locked. Retry only within the selected country.
        if not articles and category:
            articles = client.top_headlines(country=country, page_size=24)

        if not articles:
            articles = client.everything_by_country_sources(
                country=country,
                category=category,
                page_size=24,
            )

        if not articles:
            country_name = COUNTRY_NAMES.get(country, country)
            fallback_query = f"{country_name} {category or 'news'}"
            articles = client.everything(q=fallback_query, page_size=24)

        return articles

    @app.route("/", methods=["GET"])
    def index():
        error = ""
        warning = ""
        articles: List[NewsArticle] = []
        clusters: Optional[List[int]] = None
        selected_idx: Optional[int] = None
        similar_items: List[Dict[str, Any]] = []

        try:
            articles = _get_articles()

            # ✅ NEW: Save fetched articles to DB
            if articles:
                save_articles(articles)

        except NewsApiError as e:
            warning = str(e)

            # ✅ NEW: Load from DB if API fails
            db_rows = get_articles()
            if db_rows:
                articles = [
                    NewsArticle(
                        title=r[0],
                        description=r[1],
                        content=r[2],
                        url="",
                        image_url="",
                        source="DB",
                        author="",
                        published_at=""
                    )
                    for r in db_rows
                ]
            else:
                articles = FALLBACK_ARTICLES

        mode = (request.args.get("mode") or "top").strip()
        q = (request.args.get("q") or "").strip()
        if articles and mode == "search" and q:
            try:
                keep_idxs = filter_by_query(
                    titles=[a.title for a in articles],
                    descriptions=[a.description for a in articles],
                    contents=[a.content for a in articles],
                    query=q,
                )
                articles = [articles[i] for i in keep_idxs]
            except Exception:
                pass

        tfidf_ready = False
        if articles:
            _, X = build_tfidf_matrix(
                titles=[a.title for a in articles],
                descriptions=[a.description for a in articles],
                contents=[a.content for a in articles],
            )
            tfidf_ready = True

            idx_param = (request.args.get("idx") or "").strip()
            if idx_param:
                try:
                    selected_idx = int(idx_param)
                except ValueError:
                    selected_idx = None

            if selected_idx is not None and 0 <= selected_idx < len(articles):
                similar = most_similar_articles(X, article_index=selected_idx, top_k=6)
                similar_items = [
                    {
                        "idx": r.index,
                        "score": round(r.score, 3),
                        "title": articles[r.index].title,
                    }
                    for r in similar
                ]

            if (request.args.get("cluster") or "").strip() == "1":
                clusters = cluster_articles(X, n_clusters=4)

        article_dicts: List[Dict[str, Any]] = [asdict(a) for a in articles]
        selected_article = asdict(articles[selected_idx]) if selected_idx is not None and articles else None

        return render_template(
            "index.html",
            error=error,
            warning=warning,
            articles=article_dicts,
            clusters=clusters,
            tfidf_ready=tfidf_ready,
            selected_article=selected_article,
            selected_idx=selected_idx,
            similar=similar_items,
            params={
                "mode": (request.args.get("mode") or "top").strip(),
                "q": (request.args.get("q") or "").strip(),
                "country": _normalize_country(request.args.get("country") or "us"),
                "category": _normalize_category(request.args.get("category") or ""),
                "cluster": (request.args.get("cluster") or "").strip(),
            },
        )

    @app.route("/go", methods=["POST"])
    def go():
        mode = (request.form.get("mode") or "top").strip()
        q = (request.form.get("q") or "").strip()
        country = _normalize_country(request.form.get("country") or "us")
        category = _normalize_category(request.form.get("category") or "")
        cluster = "1" if (request.form.get("cluster") == "on") else ""
        if mode != "search":
            q = ""

        return redirect(
            url_for(
                "index",
                mode=mode,
                q=q,
                country=country,
                category=category,
                cluster=cluster,
            )
        )

    return app


if __name__ == "__main__":
    app = create_app()
    debug = (os.environ.get("FLASK_DEBUG") or "0").strip() in {"1", "true", "True", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug)
