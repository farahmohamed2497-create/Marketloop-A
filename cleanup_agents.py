import sqlite3

KEEP = {"rag-agent", "planning-agent", "state-graph-agent"}

conn = sqlite3.connect('marketloop.db')

all_agents = [row[0] for row in conn.execute("SELECT agent_id FROM Agents").fetchall()]
to_remove = [a for a in all_agents if a not in KEEP]

for agent_id in to_remove:
    conn.execute("DELETE FROM Agent_Tools WHERE agent_id = ?", (agent_id,))
    conn.execute("DELETE FROM Agents WHERE agent_id = ?", (agent_id,))

conn.commit()

print("Removed:", to_remove)
print("Remaining:", [row[0] for row in conn.execute("SELECT agent_id FROM Agents").fetchall()])