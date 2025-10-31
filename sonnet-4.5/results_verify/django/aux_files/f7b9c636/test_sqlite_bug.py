#!/usr/bin/env python3
import sqlite3

print("Testing SQLite QUOTE() with empty parameters...")

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

params = ()
sql = "SELECT " + ", ".join(["QUOTE(?)"] * len(params))

print(f"Generated SQL: '{sql}'")
print(f"Parameters: {params}")

try:
    result = cursor.execute(sql, params).fetchone()
    print(f"Result: {result}")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")

conn.close()