import sqlite3

# Test with SQLite
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a simple table
cursor.execute('CREATE TABLE test (id INTEGER)')
cursor.execute('INSERT INTO test VALUES (1), (2), (3), (4), (5)')

# Try negative LIMIT
try:
    cursor.execute('SELECT * FROM test LIMIT -5 OFFSET 2')
    print(f"SQLite result: {cursor.fetchall()}")
except Exception as e:
    print(f"SQLite error: {e}")

conn.close()