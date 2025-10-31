#!/usr/bin/env python3
"""Test Django SQLite LPAD/RPAD functions with negative lengths"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

print("=== Testing Django SQLite LPAD/RPAD Functions ===\n")

# Test the specific failing inputs mentioned in the bug report
print("1. Testing specific failing inputs from bug report:")
print(f"_sqlite_lpad('00', -1, '0') = {repr(_sqlite_lpad('00', -1, '0'))}")
print(f"Expected: ''")
print(f"_sqlite_rpad('hello', -2, 'X') = {repr(_sqlite_rpad('hello', -2, 'X'))}")
print(f"Expected: ''\n")

# Test more cases
print("2. Testing additional negative length cases:")
test_cases = [
    ('test', -1, 'x'),
    ('abcdef', -3, 'y'),
    ('', -5, 'z'),
    ('single', -10, 'pad'),
]

for text, length, fill in test_cases:
    lpad_result = _sqlite_lpad(text, length, fill)
    rpad_result = _sqlite_rpad(text, length, fill)
    print(f"LPAD({text!r}, {length}, {fill!r}) = {lpad_result!r}")
    print(f"RPAD({text!r}, {length}, {fill!r}) = {rpad_result!r}")

print("\n3. Testing zero length:")
print(f"_sqlite_lpad('test', 0, 'x') = {repr(_sqlite_lpad('test', 0, 'x'))}")
print(f"_sqlite_rpad('test', 0, 'x') = {repr(_sqlite_rpad('test', 0, 'x'))}")

print("\n4. Testing positive length (for comparison):")
print(f"_sqlite_lpad('test', 6, 'x') = {repr(_sqlite_lpad('test', 6, 'x'))}")
print(f"_sqlite_rpad('test', 6, 'x') = {repr(_sqlite_rpad('test', 6, 'x'))}")

print("\n5. Testing NULL handling:")
print(f"_sqlite_lpad(None, 5, 'x') = {repr(_sqlite_lpad(None, 5, 'x'))}")
print(f"_sqlite_lpad('test', None, 'x') = {repr(_sqlite_lpad('test', None, 'x'))}")
print(f"_sqlite_lpad('test', 5, None) = {repr(_sqlite_lpad('test', 5, None))}")

# Run the property-based tests
print("\n6. Running property-based tests from bug report:")

failed_lpad = []
failed_rpad = []

@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_lpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    if result != "":
        failed_lpad.append((text, length, fill_text, result))
        return False
    return True

@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_rpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_rpad(text, length, fill_text)
    if result != "":
        failed_rpad.append((text, length, fill_text, result))
        return False
    return True

try:
    test_lpad_negative_length_returns_empty()
    print("LPAD property test would PASS if empty string was returned")
except AssertionError:
    print(f"LPAD property test FAILS - Found {len(failed_lpad)} failures")
    if failed_lpad:
        print(f"  Example failure: LPAD{failed_lpad[0][:3]} returned {failed_lpad[0][3]!r}")

try:
    test_rpad_negative_length_returns_empty()
    print("RPAD property test would PASS if empty string was returned")
except AssertionError:
    print(f"RPAD property test FAILS - Found {len(failed_rpad)} failures")
    if failed_rpad:
        print(f"  Example failure: RPAD{failed_rpad[0][:3]} returned {failed_rpad[0][3]!r}")

print("\n7. Understanding the current behavior:")
print("When length is negative, Python's slicing behavior text[:length] is used.")
print("For example:")
print(f"  'hello'[:-2] = {'hello'[:-2]!r} (removes last 2 chars)")
print(f"  '00'[:-1] = {'00'[:-1]!r} (removes last char)")
print("This explains why negative lengths return truncated strings instead of empty strings.")