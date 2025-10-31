#!/usr/bin/env python3
"""Test Django SQLite LPAD/RPAD functions with negative lengths"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

print("=== Testing Django SQLite LPAD/RPAD Functions ===\n")

# Test the specific failing inputs mentioned in the bug report
print("1. Testing specific failing inputs from bug report:")
result1 = _sqlite_lpad('00', -1, '0')
print(f"_sqlite_lpad('00', -1, '0') = {repr(result1)}")
print(f"Expected: ''")
print(f"Bug claim: Should return empty string, actually returns: {repr(result1)}")
print()

result2 = _sqlite_rpad('hello', -2, 'X')
print(f"_sqlite_rpad('hello', -2, 'X') = {repr(result2)}")
print(f"Expected: ''")
print(f"Bug claim: Should return empty string, actually returns: {repr(result2)}")
print()

# Test more cases
print("2. Testing additional negative length cases:")
test_cases = [
    ('test', -1, 'x'),
    ('abcdef', -3, 'y'),
    ('', -5, 'z'),
    ('single', -10, 'pad'),
    ('a', -1, 'b'),
    ('abc', -100, 'x'),
]

for text, length, fill in test_cases:
    lpad_result = _sqlite_lpad(text, length, fill)
    rpad_result = _sqlite_rpad(text, length, fill)
    print(f"LPAD({text!r:8}, {length:4}, {fill!r:5}) = {lpad_result!r:10} (expected: '')")
    print(f"RPAD({text!r:8}, {length:4}, {fill!r:5}) = {rpad_result!r:10} (expected: '')")

print("\n3. Testing zero length:")
zero_lpad = _sqlite_lpad('test', 0, 'x')
zero_rpad = _sqlite_rpad('test', 0, 'x')
print(f"_sqlite_lpad('test', 0, 'x') = {repr(zero_lpad)} (returns empty as expected)")
print(f"_sqlite_rpad('test', 0, 'x') = {repr(zero_rpad)} (returns empty as expected)")

print("\n4. Testing positive length (for comparison - these work correctly):")
print(f"_sqlite_lpad('test', 6, 'x') = {repr(_sqlite_lpad('test', 6, 'x'))}")
print(f"_sqlite_rpad('test', 6, 'x') = {repr(_sqlite_rpad('test', 6, 'x'))}")

print("\n5. Testing NULL handling (these work correctly):")
print(f"_sqlite_lpad(None, 5, 'x') = {repr(_sqlite_lpad(None, 5, 'x'))}")
print(f"_sqlite_lpad('test', None, 'x') = {repr(_sqlite_lpad('test', None, 'x'))}")
print(f"_sqlite_lpad('test', 5, None) = {repr(_sqlite_lpad('test', 5, None))}")

print("\n6. Understanding the current implementation bug:")
print("The current implementation uses Python's negative slicing:")
print("  - text[:length] when length is negative")
print("  - This causes 'hello'[:-2] = 'hel' (removes last 2 chars)")
print("  - This causes '00'[:-1] = '0' (removes last char)")
print("\n7. Expected behavior (per PostgreSQL standard):")
print("  - Negative length should return empty string ''")
print("  - This is what PostgreSQL and Redshift do")
print("\n8. Alternative behavior (MySQL/MariaDB):")
print("  - Negative length returns NULL")
print("  - Django chose neither standard, using Python slicing instead")

# Verify the bug exists
print("\n9. Bug verification:")
failures = 0
for text, length, fill in [('00', -1, '0'), ('hello', -2, 'X'), ('test', -1, 'x')]:
    lpad_result = _sqlite_lpad(text, length, fill)
    rpad_result = _sqlite_rpad(text, length, fill)
    if lpad_result != '':
        failures += 1
        print(f"FAIL: LPAD({text!r}, {length}, {fill!r}) returned {lpad_result!r} instead of ''")
    if rpad_result != '':
        failures += 1
        print(f"FAIL: RPAD({text!r}, {length}, {fill!r}) returned {rpad_result!r} instead of ''")

if failures > 0:
    print(f"\n✗ Bug confirmed: {failures} test failures")
else:
    print("\n✓ No bug found (all tests passed)")