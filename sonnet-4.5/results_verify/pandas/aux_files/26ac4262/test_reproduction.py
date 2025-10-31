#!/usr/bin/env python3
"""Test to reproduce the bug in pandas.io.sas.sas_xport._split_line"""

from pandas.io.sas.sas_xport import _split_line

print("Testing _split_line function without '_' field in parts...")
print("=" * 50)

parts = [("name", 10), ("age", 5)]
s = "John Doe  30   "

print(f"Input string: '{s}'")
print(f"Parts: {parts}")
print()

try:
    result = _split_line(s, parts)
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Full error message: {repr(e)}")

print()
print("=" * 50)
print("Testing _split_line function WITH '_' field in parts...")

parts_with_underscore = [("name", 10), ("_", 2), ("age", 3)]
s2 = "John Doe  --30 "

print(f"Input string: '{s2}'")
print(f"Parts: {parts_with_underscore}")
print()

try:
    result2 = _split_line(s2, parts_with_underscore)
    print(f"Result: {result2}")
except KeyError as e:
    print(f"KeyError raised: {e}")