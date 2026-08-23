import sqlite3

conn = sqlite3.connect('marketloop.db')
conn.execute("DELETE FROM Agent_Tools")
conn.execute("DELETE FROM Agents")
conn.commit()
print("Cleared. Remaining agents:", conn.execute("SELECT * FROM Agents").fetchall())