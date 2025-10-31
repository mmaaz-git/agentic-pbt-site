#!/usr/bin/env python3
"""
Minimal reproduction of the truncate_name bug in Django.
This demonstrates that the function violates its contract when length < hash_len.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.utils import truncate_name, split_identifier

# Test case from the bug report
identifier = '00'
length = 1

print(f"Testing truncate_name('{identifier}', length={length})")
print(f"Expected: Result with length <= {length}")

result = truncate_name(identifier, length=length)
namespace, name = split_identifier(result)

print(f"Actual result: '{result}'")
print(f"Name part: '{name}' (length = {len(name)})")

if len(name) > length:
    print(f"\n❌ BUG CONFIRMED: The function returns a name of length {len(name)}, which exceeds the requested length of {length}")
else:
    print(f"\n✓ OK: Result respects the length constraint")

# Additional test cases to demonstrate the bug pattern
print("\n" + "="*60)
print("Additional test cases:")
print("="*60)

test_cases = [
    ('abc', 1),
    ('test', 2),
    ('database_name', 3),
    ('very_long_identifier_name', 4),
    ('short', 5),
]

for test_id, test_length in test_cases:
    result = truncate_name(test_id, length=test_length)
    _, name_part = split_identifier(result)
    status = "❌ BUG" if len(name_part) > test_length else "✓ OK"
    print(f"truncate_name('{test_id}', {test_length}) -> '{result}' (name length={len(name_part)}) {status}")