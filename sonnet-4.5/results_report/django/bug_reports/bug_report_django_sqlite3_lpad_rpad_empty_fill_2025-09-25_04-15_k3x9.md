# Bug Report: Django SQLite3 LPAD/RPAD Empty Fill Text

**Target**: `django.db.backends.sqlite3._functions._sqlite_lpad` and `_sqlite_rpad`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_lpad` and `_sqlite_rpad` functions return strings of incorrect length when `fill_text` is an empty string, violating the invariant that these functions should always return a string of exactly the specified `length`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad


@given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
@example(text="hello", length=10)
def test_lpad_with_empty_fill_text_violates_length(text, length):
    fill_text = ""
    result = _sqlite_lpad(text, length, fill_text)

    if result is not None:
        assert len(result) == length, (
            f"lpad with empty fill_text should still return exact length. "
            f"Expected length {length}, got {len(result)}: {result!r}"
        )


@given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
@example(text="hello", length=10)
def test_rpad_with_empty_fill_text_violates_length(text, length):
    fill_text = ""
    result = _sqlite_rpad(text, length, fill_text)

    if result is not None:
        assert len(result) == length, (
            f"rpad with empty fill_text should still return exact length. "
            f"Expected length {length}, got {len(result)}: {result!r}"
        )
```

**Failing input**: `text='hello', length=10, fill_text=''`

## Reproducing the Bug

```python
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

text = "hello"
length = 10
fill_text = ""

result_lpad = _sqlite_lpad(text, length, fill_text)
print(f"LPAD result: {result_lpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_lpad)}")

result_rpad = _sqlite_rpad(text, length, fill_text)
print(f"RPAD result: {result_rpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_rpad)}")
```

Output:
```
LPAD result: 'hello'
Expected length: 10, Actual length: 5

RPAD result: 'hello'
Expected length: 10, Actual length: 5
```

## Why This Is A Bug

SQL `LPAD` and `RPAD` functions are expected to always return a string of exactly the specified length. This is a fundamental invariant of these functions. When the padding string is empty and the text is shorter than the target length, standard SQL behavior (as implemented in PostgreSQL, MySQL, and Oracle) is to return `NULL` because it's impossible to pad to the required length with an empty string.

Django's implementation violates this invariant by returning the original text unchanged, which has the wrong length. This could lead to:
1. Data corruption when the result is expected to be exactly `length` characters
2. SQL query errors if the result is used in contexts expecting fixed-width strings
3. Silent failures in formatting operations

## Fix

Add a check for empty `fill_text` at the beginning of both functions:

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -389,6 +389,8 @@ def _sqlite_lpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    if not fill_text:
+        return None
     delta = length - len(text)
     if delta <= 0:
         return text[:length]
@@ -438,6 +440,8 @@ def _sqlite_rpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    if not fill_text:
+        return None
     return (text + fill_text * length)[:length]
```