import sqlite3
conn = sqlite3.connect('marketloop.db')
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print(tables)