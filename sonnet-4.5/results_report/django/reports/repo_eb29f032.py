import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', DEFAULT_CHARSET='utf-8')
    django.setup()

from django.http import QueryDict

# Create a QueryDict with a key-value pair
qd = QueryDict(mutable=True)
qd['key'] = 'value'

# Access via __getitem__ (qd[key])
print(f"qd['key'] = {qd['key']!r}")
print(f"Type: {type(qd['key']).__name__}")

# Create another QueryDict for pop() test
qd2 = QueryDict(mutable=True)
qd2['key'] = 'value'

# Access via pop()
popped = qd2.pop('key')
print(f"\nqd.pop('key') = {popped!r}")
print(f"Type: {type(popped).__name__}")

# Demonstrate the inconsistency
print("\nThese should be the same type, but they're not!")
print(f"__getitem__ returns: {type(qd['key']).__name__}")
print(f"pop() returns: {type(popped).__name__}")

# Show what happens with multiple values
qd3 = QueryDict(mutable=True)
qd3.setlist('multi', ['first', 'second', 'third'])

print(f"\nFor multiple values:")
print(f"qd3['multi'] (via __getitem__) = {qd3['multi']!r}")
print(f"Type: {type(qd3['multi']).__name__}")

qd4 = QueryDict(mutable=True)
qd4.setlist('multi', ['first', 'second', 'third'])
multi_popped = qd4.pop('multi')
print(f"\nqd4.pop('multi') = {multi_popped!r}")
print(f"Type: {type(multi_popped).__name__}")

print("\nThe issue: __getitem__ returns the last value (string), but pop() returns the entire list!")