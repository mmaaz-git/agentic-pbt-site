import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from django.http import parse_cookie

# Test case 1: Cookie with quoted value containing semicolon
cookie_string = 'session="abc;123"; user=john'
result = parse_cookie(cookie_string)

print(f"Input:    {cookie_string!r}")
print(f"Result:   {result}")
print(f"Expected: {{'session': 'abc;123', 'user': 'john'}}")
print()

# Test case 2: Simple semicolon in value
cookie_string2 = '0=;'
result2 = parse_cookie(cookie_string2)

print(f"Input:    {cookie_string2!r}")
print(f"Result:   {result2}")
print(f"Expected: {{'0': ';'}}")
print()

# Compare with Python's SimpleCookie
from http.cookies import SimpleCookie

print("Python's SimpleCookie behavior:")
sc = SimpleCookie()
sc.load('session="abc;123"; user=john')
print(f"SimpleCookie result: {dict((k, v.value) for k, v in sc.items())}")

# Verify the assertion failure
try:
    assert result == {'session': 'abc;123', 'user': 'john'}
    print("Assertion passed")
except AssertionError:
    print("AssertionError: parse_cookie incorrectly handled quoted semicolons")