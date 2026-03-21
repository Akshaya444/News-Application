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


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")

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
        country = (request.args.get("country") or "us").strip()
        category = (request.args.get("category") or "").strip()

        # Strategy:
        # - First try the "correct" endpoint based on mode.
        # - If it returns empty, retry with less constraints so the UI isn't empty.
        if mode == "search":
            if not q:
                return []

            articles = client.everything(q=q, page_size=24)
            if not articles:
                # Fallback to top-headlines with the same query.
                articles = client.top_headlines(country=country, category=category, q=q, page_size=24)
            return articles

        # Top headlines
        if q:
            articles = client.top_headlines(country=country, category=category, q=q, page_size=24)
else:
    articles = client.top_headlines(country=country, category=category, page_size=24)
        if not articles and q:
            # If keyword filtering returns nothing, show top headlines for the same country/category.
            articles = client.top_headlines(country=country, category=category, q="", page_size=24)
        if not articles and q:
            # Last fallback: search across news by keyword.
            articles = client.everything(q=q, page_size=24)
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
        except NewsApiError as e:
            # Keep the app usable even if NewsAPI is temporarily unreachable.
            warning = str(e)
            articles = FALLBACK_ARTICLES

        # Keyword-based filtering: keep only articles relevant to what the user typed.
        q = (request.args.get("q") or "").strip()
        if articles and q:
            try:
                keep_idxs = filter_by_query(
                    titles=[a.title for a in articles],
                    descriptions=[a.description for a in articles],
                    contents=[a.content for a in articles],
                    query=q,
                )
                articles = [articles[i] for i in keep_idxs]
            except Exception:
                # If the model is missing or something goes wrong,
                # just show the raw NewsAPI results.
                pass

        # ML: build TF-IDF and optionally cluster this page of results
        tfidf_ready = False
        if articles:
            _, X = build_tfidf_matrix(
                titles=[a.title for a in articles],
                descriptions=[a.description for a in articles],
                contents=[a.content for a in articles],
            )
            tfidf_ready = True

            # If idx is provided, show article detail + similarity list on the same page.
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

        # Store minimal article dicts in template (no server-side session needed)
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
                "country": (request.args.get("country") or "us").strip(),
                "category": (request.args.get("category") or "").strip(),
                "cluster": (request.args.get("cluster") or "").strip(),
            },
        )

    @app.route("/go", methods=["POST"])
    def go():
        # Small helper to keep the form POST -> redirect with querystring
        mode = (request.form.get("mode") or "top").strip()
        q = (request.form.get("q") or "").strip()
        country = (request.form.get("country") or "us").strip()
        category = (request.form.get("category") or "").strip()
        cluster = "1" if (request.form.get("cluster") == "on") else ""

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
