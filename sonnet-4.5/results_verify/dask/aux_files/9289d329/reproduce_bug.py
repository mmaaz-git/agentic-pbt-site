import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta

print("Addition returns:", type(_sqlite_format_dtdelta("+", 1000000, 2000000)))
print("Subtraction returns:", type(_sqlite_format_dtdelta("-", 2000000, 1000000)))
print("Multiplication returns:", type(_sqlite_format_dtdelta("*", 2.5, 3.0)))
print("Division returns:", type(_sqlite_format_dtdelta("/", 6.0, 2.0)))

# Also check the actual values
print("\nActual values:")
print("Addition:", _sqlite_format_dtdelta("+", 1000000, 2000000))
print("Subtraction:", _sqlite_format_dtdelta("-", 2000000, 1000000))
print("Multiplication:", _sqlite_format_dtdelta("*", 2.5, 3.0))
print("Division:", _sqlite_format_dtdelta("/", 6.0, 2.0))