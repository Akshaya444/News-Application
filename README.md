# News Application Using NewsAPI (Flask + ML)

## Features
- Top headlines + keyword search (powered by NewsAPI)
- Article detail page
- ML: “Similar articles” recommendations using TF‑IDF + cosine similarity
- Optional ML: clustering of current results (KMeans) for quick grouping

## Setup
1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (copy from `.env.example`) and set your NewsAPI key:

```env
NEWSAPI_KEY=your_key_here
FLASK_DEBUG=1
```

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes
- NewsAPI requires an API key from [NewsAPI.org](https://newsapi.org/).
- Free keys may have limitations (sources, rate limits, CORS). This server-side Flask app avoids browser CORS issues.
