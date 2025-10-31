#!/usr/bin/env python3
"""Test the parse_cookie bug report - reproduction steps"""

import os
import sys

# Configure Django
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

# Import the functions we need
from django.http import parse_cookie
from http.cookies import SimpleCookie

# Test 1: Property-based test with simple case
print("=" * 60)
print("Test 1: Property-based test failure case")
print("=" * 60)

cookie_dict = {'0': ';'}
cookie_string = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
print(f"Input dict: {cookie_dict}")
print(f"Cookie string: {cookie_string!r}")

parsed = parse_cookie(cookie_string)
print(f"Parsed result: {parsed}")
print(f"Expected: {cookie_dict}")

try:
    assert parsed == cookie_dict
    print("✓ Test passed")
except AssertionError:
    print("✗ Test failed: parsed result doesn't match expected")

# Test 2: Quoted cookie values with semicolons
print("\n" + "=" * 60)
print("Test 2: Quoted cookie values with semicolons")
print("=" * 60)

cookie_string = 'session="abc;123"; user=john'
print(f"Input:    {cookie_string!r}")

result = parse_cookie(cookie_string)
print(f"Result:   {result}")

expected = {'session': 'abc;123', 'user': 'john'}
print(f"Expected: {expected}")

try:
    assert result == expected
    print("✓ Test passed")
except AssertionError:
    print("✗ Test failed: result doesn't match expected")

# Test 3: Compare with Python's SimpleCookie
print("\n" + "=" * 60)
print("Test 3: Compare with Python's SimpleCookie")
print("=" * 60)

cookie_string = 'session="abc;123"; user=john'
print(f"Cookie string: {cookie_string!r}")

# Django's parse_cookie
django_result = parse_cookie(cookie_string)
print(f"Django parse_cookie: {django_result}")

# Python's SimpleCookie
sc = SimpleCookie()
sc.load(cookie_string)
python_result = {key: morsel.value for key, morsel in sc.items()}
print(f"Python SimpleCookie: {python_result}")

if django_result == python_result:
    print("✓ Results match")
else:
    print("✗ Results differ")

# Test 4: More complex examples
print("\n" + "=" * 60)
print("Test 4: Additional test cases")
print("=" * 60)

test_cases = [
    ('key=value', {'key': 'value'}, "Simple cookie"),
    ('key="value"', {'key': 'value'}, "Quoted value"),
    ('key="val;ue"', {'key': 'val;ue'}, "Quoted value with semicolon"),
    ('a=1; b=2', {'a': '1', 'b': '2'}, "Multiple cookies"),
    ('a="1;2"; b=3', {'a': '1;2', 'b': '3'}, "Mixed quoted and unquoted"),
    ('name="a=b;c=d"', {'name': 'a=b;c=d'}, "Quoted value with = and ;"),
]

for cookie_str, expected, description in test_cases:
    print(f"\nTest: {description}")
    print(f"  Input:    {cookie_str!r}")
    result = parse_cookie(cookie_str)
    print(f"  Result:   {result}")
    print(f"  Expected: {expected}")
    if result == expected:
        print(f"  ✓ Pass")
    else:
        print(f"  ✗ Fail")

print("\n" + "=" * 60)
print("Summary: The bug exists - parse_cookie incorrectly handles")
print("quoted values containing semicolons.")
print("=" * 60)