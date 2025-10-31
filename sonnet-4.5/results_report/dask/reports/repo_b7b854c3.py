import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

print("Testing _sqlite_lpad with negative length:")
print(f"_sqlite_lpad('00', -1, '0') = {repr(_sqlite_lpad('00', -1, '0'))}")
print(f"Expected: ''")
print()

print("Testing _sqlite_rpad with negative length:")
print(f"_sqlite_rpad('hello', -2, 'X') = {repr(_sqlite_rpad('hello', -2, 'X'))}")
print(f"Expected: ''")
print()

print("Additional test cases:")
print(f"_sqlite_lpad('test', -3, '*') = {repr(_sqlite_lpad('test', -3, '*'))}")
print(f"_sqlite_rpad('example', -5, '#') = {repr(_sqlite_rpad('example', -5, '#'))}")