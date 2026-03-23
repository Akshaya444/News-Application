import sqlite3

DB_NAME = "news.db"

def get_connection():
    return sqlite3.connect(DB_NAME)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_articles(articles):
    conn = get_connection()
    cursor = conn.cursor()

    for a in articles:
        cursor.execute("""
        INSERT INTO articles (title, description, content)
        VALUES (?, ?, ?)
        """, (a.title, a.description, a.content))

    conn.commit()
    conn.close()


def get_articles():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT title, description, content FROM articles ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()

    conn.close()
    return rows
