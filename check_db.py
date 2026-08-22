import sqlite3

conn = sqlite3.connect("marketloop.db")

tables = conn.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

for (table_name,) in tables:
    print(f"\n--- {table_name} ---")

    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()

    print(schema[0] if schema else "No schema found")

conn.close()