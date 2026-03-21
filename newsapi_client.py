from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class NewsArticle:
    title: str
    description: str
    content: str
    url: str
    image_url: str
    source: str
    author: str
    published_at: str

    @property
    def published_at_dt(self) -> Optional[datetime]:
        if not self.published_at:
            return None
        # NewsAPI uses ISO 8601 timestamps e.g. "2025-01-01T10:00:00Z"
        try:
            return datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
        except ValueError:
            return None


class NewsApiError(RuntimeError):
    pass


class NewsApiClient:
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2") -> None:
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("Missing NEWSAPI_KEY")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def top_headlines(
        self,
        *,
        country: str = "us",
        category: str = "",
        q: str = "",
        page_size: int = 20,
        page: int = 1,
    ) -> List[NewsArticle]:
        params: Dict[str, Any] = {
            "country": country,
            "pageSize": page_size,
            "page": page,
        }
        if category:
            params["category"] = category
        if q:
            params["q"] = q
        return self._get("/top-headlines", params=params)

    def everything(
        self,
        *,
        q: str,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
        page: int = 1,
    ) -> List[NewsArticle]:
        params: Dict[str, Any] = {
            "q": q,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page,
        }
        return self._get("/everything", params=params)

    def _get(self, path: str, params: Dict[str, Any]) -> List[NewsArticle]:
        url = f"{self.base_url}{path}"
        headers = {"X-Api-Key": self.api_key}
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=25)
        except requests.RequestException as e:
            raise NewsApiError("Network error calling NewsAPI. Please check your internet/firewall and try again.") from e

        try:
            payload = resp.json()
        except ValueError as e:
            raise NewsApiError(f"Invalid JSON from NewsAPI (status {resp.status_code})") from e

        if resp.status_code != 200:
            message = payload.get("message") if isinstance(payload, dict) else None
            raise NewsApiError(message or f"NewsAPI error (status {resp.status_code})")

        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise NewsApiError("Unexpected response from NewsAPI")

        raw_articles = payload.get("articles") or []
        return [self._normalize_article(a) for a in raw_articles if isinstance(a, dict)]

    def _normalize_article(self, a: Dict[str, Any]) -> NewsArticle:
        source = a.get("source") or {}
        return NewsArticle(
            title=(a.get("title") or "").strip(),
            description=(a.get("description") or "").strip(),
            content=(a.get("content") or "").strip(),
            url=(a.get("url") or "").strip(),
            image_url=(a.get("urlToImage") or "").strip(),
            source=(source.get("name") or "").strip() if isinstance(source, dict) else "",
            author=(a.get("author") or "").strip(),
            published_at=(a.get("publishedAt") or "").strip(),
        )
