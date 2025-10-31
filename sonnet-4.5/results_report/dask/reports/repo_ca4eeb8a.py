#!/usr/bin/env python3
"""
Minimal reproduction of the Django SQLite format_dtdelta bug.
Demonstrates inconsistent return types: str for +/- but float for *// operations.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta

print("Testing _sqlite_format_dtdelta return types:\n")

# Test addition
result_add = _sqlite_format_dtdelta("+", 1000000, 2000000)
print(f"Addition (+):")
print(f"  Input: _sqlite_format_dtdelta('+', 1000000, 2000000)")
print(f"  Result: {result_add!r}")
print(f"  Type: {type(result_add)}")
print()

# Test subtraction
result_sub = _sqlite_format_dtdelta("-", 2000000, 1000000)
print(f"Subtraction (-):")
print(f"  Input: _sqlite_format_dtdelta('-', 2000000, 1000000)")
print(f"  Result: {result_sub!r}")
print(f"  Type: {type(result_sub)}")
print()

# Test multiplication
result_mul = _sqlite_format_dtdelta("*", 2.5, 3.0)
print(f"Multiplication (*):")
print(f"  Input: _sqlite_format_dtdelta('*', 2.5, 3.0)")
print(f"  Result: {result_mul!r}")
print(f"  Type: {type(result_mul)}")
print()

# Test division
result_div = _sqlite_format_dtdelta("/", 6.0, 2.0)
print(f"Division (/):")
print(f"  Input: _sqlite_format_dtdelta('/', 6.0, 2.0)")
print(f"  Result: {result_div!r}")
print(f"  Type: {type(result_div)}")
print()

# Summary
print("=" * 50)
print("SUMMARY:")
print("=" * 50)
print(f"Addition returns:       {type(result_add).__name__}")
print(f"Subtraction returns:    {type(result_sub).__name__}")
print(f"Multiplication returns: {type(result_mul).__name__}")
print(f"Division returns:       {type(result_div).__name__}")
print()
print("Expected: All operations should return 'str' type")
print("Actual:   Addition and subtraction return 'str',")
print("          but multiplication and division return 'float'")