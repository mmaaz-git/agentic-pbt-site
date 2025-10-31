# Bug Report: django.utils.http.quote_etag Violates Idempotence with Quote Characters

**Target**: `django.utils.http.quote_etag`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `quote_etag` function violates its documented idempotence property when given input strings containing quote characters, producing different results when called multiple times on the same input and generating ETags that violate RFC 9110.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.http import quote_etag


@given(st.text())
def test_quote_etag_idempotence(s):
    result1 = quote_etag(s)
    result2 = quote_etag(result1)
    assert result1 == result2, f"quote_etag is not idempotent: quote_etag({s!r}) = {result1!r}, but quote_etag({result1!r}) = {result2!r}"

# Run the test
if __name__ == "__main__":
    test_quote_etag_idempotence()
```

<details>

<summary>
**Failing input**: `'"'` (a single quote character)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 13, in <module>
    test_quote_etag_idempotence()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 6, in test_quote_etag_idempotence
    def test_quote_etag_idempotence(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 9, in test_quote_etag_idempotence
    assert result1 == result2, f"quote_etag is not idempotent: quote_etag({s!r}) = {result1!r}, but quote_etag({result1!r}) = {result2!r}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: quote_etag is not idempotent: quote_etag('"') = '"""', but quote_etag('"""') = '"""""'
Falsifying example: test_quote_etag_idempotence(
    s='"',
)
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
AssertionError: quote_etag is not idempotent
</summary>
```
quote_etag('"') = '"""'
quote_etag('"""') = '"""""'

Are they equal? False
AssertionError: quote_etag is not idempotent
  First call:  '"' -> '"""'
  Second call: '"""' -> '"""""'
```
</details>

## Why This Is A Bug

The function's docstring explicitly promises: "If the provided string is already a quoted ETag, return it. Otherwise, wrap the string in quotes, making it a strong ETag." This implies idempotence - calling `quote_etag` on its own output should return the same value.

The bug occurs because:

1. **The ETAG_MATCH regex pattern** at line 15-25 of django/utils/http.py only matches quoted strings that don't contain internal quotes: `"[^"]*"`
2. **When `quote_etag` wraps a string containing quotes** (like `"`), it produces `"""` which doesn't match the ETAG_MATCH pattern
3. **Subsequent calls wrap it again**, producing `"""""`, then `"""""""`, and so on, violating idempotence

Additionally, this violates **RFC 9110 Section 8.8.3** which explicitly disallows quote characters within ETags. The ABNF grammar specifies:
- `opaque-tag = DQUOTE *etagc DQUOTE`
- Where `etagc = %x21 / %x23-7E / obs-text`
- Double quotes (`%x22`) are explicitly excluded from allowed characters

This could cause issues in production where `quote_etag` might be called multiple times through middleware layers or caching systems, resulting in incorrectly formatted ETags with multiple layers of quotes.

## Relevant Context

The function is located in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/utils/http.py` at lines 212-220:

```python
def quote_etag(etag_str):
    """
    If the provided string is already a quoted ETag, return it. Otherwise, wrap
    the string in quotes, making it a strong ETag.
    """
    if ETAG_MATCH.match(etag_str):
        return etag_str
    else:
        return '"%s"' % etag_str
```

The ETAG_MATCH regex (lines 15-25) correctly implements RFC 9110's requirements by only matching ETags without internal quotes. However, the function doesn't validate or escape input that contains quotes, leading to the creation of invalid ETags.

## Proposed Fix

The function should validate input and either escape quotes or reject invalid input. Here's a fix that validates and rejects invalid input:

```diff
def quote_etag(etag_str):
    """
    If the provided string is already a quoted ETag, return it. Otherwise, wrap
    the string in quotes, making it a strong ETag.
    """
    if ETAG_MATCH.match(etag_str):
        return etag_str
    else:
+       # RFC 9110 doesn't allow quotes in ETag content
+       if '"' in etag_str:
+           raise ValueError(f"Invalid ETag content: quotes are not allowed in ETags per RFC 9110")
        return '"%s"' % etag_str
```