#!/usr/bin/env python3
"""Test how Python's standard library handles cookies with semicolons"""

from http.cookies import SimpleCookie
import http.cookies

print("Testing Python's standard library cookie handling")
print("=" * 60)

# Test 1: Quoted value with semicolon
print("\nTest 1: Quoted value with semicolon")
cookie_string = 'session="abc;123"; user=john'
print(f"Input: {cookie_string}")

sc = SimpleCookie()
sc.load(cookie_string)
print(f"SimpleCookie parsed:")
for key, morsel in sc.items():
    print(f"  {key}: value={morsel.value!r}, coded_value={morsel.coded_value!r}")

# Test 2: Check what happens with raw semicolon
print("\nTest 2: Unquoted semicolon (should fail)")
cookie_string = 'session=abc;123'
print(f"Input: {cookie_string}")

sc = SimpleCookie()
sc.load(cookie_string)
print(f"SimpleCookie parsed:")
for key, morsel in sc.items():
    print(f"  {key}: value={morsel.value!r}, coded_value={morsel.coded_value!r}")

# Test 3: Check _unquote function
print("\nTest 3: Testing cookies._unquote function")
test_values = [
    '"abc;123"',
    'abc;123',
    '"abc"',
    'abc',
    '""',
    '',
]

for val in test_values:
    unquoted = http.cookies._unquote(val)
    print(f"  _unquote({val!r}) = {unquoted!r}")

# Test 4: How SimpleCookie handles various formats
print("\nTest 4: Various cookie formats")
test_cases = [
    'key="value"',
    'key="val;ue"',
    'key="val=ue"',
    'key="val,ue"',
    'key="val ue"',
    'key=value',
    'a="1"; b="2"',
    'a="1;2"; b="3"',
]

for cookie_str in test_cases:
    print(f"\nInput: {cookie_str!r}")
    sc = SimpleCookie()
    sc.load(cookie_str)
    result = {key: morsel.value for key, morsel in sc.items()}
    print(f"Result: {result}")

# Test 5: RFC 6265 compliant parsing
print("\n" + "=" * 60)
print("RFC 6265 Analysis:")
print("According to RFC 6265, semicolons are NOT allowed in cookie values,")
print("even when quoted. The RFC states that cookie-value can be:")
print("  cookie-value = *cookie-octet / ( DQUOTE *cookie-octet DQUOTE )")
print("where cookie-octet explicitly EXCLUDES semicolons.")
print("\nSo the question is: does Python's SimpleCookie follow RFC 6265")
print("or an older RFC (like RFC 2109) that allowed quoted semicolons?")