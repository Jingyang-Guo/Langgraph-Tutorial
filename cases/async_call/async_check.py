import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("async.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

cursor = conn.cursor()
cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
threads = cursor.fetchall()
print(f"Threads in DB: {threads}")

config = {"configurable": {"thread_id": "5"}}
data = checkpointer.get(config)

for message in data["channel_values"]["messages"]:
    message.pretty_print()

conn.close()
