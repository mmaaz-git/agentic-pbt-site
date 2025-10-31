from django.utils.http import quote_etag

# Test case with a single quote character
s = '"'
result1 = quote_etag(s)
result2 = quote_etag(result1)

print(f"quote_etag({s!r}) = {result1!r}")
print(f"quote_etag({result1!r}) = {result2!r}")
print()
print(f"Are they equal? {result1 == result2}")

# The assertion that fails
try:
    assert result1 == result2
    print("Assertion passed: quote_etag is idempotent")
except AssertionError:
    print("AssertionError: quote_etag is not idempotent")
    print(f"  First call:  {s!r} -> {result1!r}")
    print(f"  Second call: {result1!r} -> {result2!r}")