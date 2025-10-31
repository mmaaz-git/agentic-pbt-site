# Bug Report: Django SQLite3 LPAD/RPAD Functions Return Incorrect Length with Empty Padding

**Target**: `django.db.backends.sqlite3._functions._sqlite_lpad` and `_sqlite_rpad`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_lpad` and `_sqlite_rpad` functions in Django's SQLite3 backend return strings with incorrect length when an empty string is used as the padding parameter, violating the fundamental SQL invariant that these functions must return strings of exactly the specified length or NULL.

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


if __name__ == "__main__":
    print("Running Hypothesis tests for LPAD/RPAD with empty fill_text...")
    print("=" * 60)

    try:
        test_lpad_with_empty_fill_text_violates_length()
        print("LPAD test passed")
    except AssertionError as e:
        print(f"LPAD test FAILED: {e}")

    try:
        test_rpad_with_empty_fill_text_violates_length()
        print("RPAD test passed")
    except AssertionError as e:
        print(f"RPAD test FAILED: {e}")
```

<details>

<summary>
**Failing input**: `text='hello', length=10`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/24
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 2 items

hypo.py::test_lpad_with_empty_fill_text_violates_length FAILED           [ 50%]
hypo.py::test_rpad_with_empty_fill_text_violates_length FAILED           [100%]

=================================== FAILURES ===================================
________________ test_lpad_with_empty_fill_text_violates_length ________________

    @given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
>   @example(text="hello", length=10)
                   ^^^

hypo.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

text = 'hello', length = 10

    @given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
    @example(text="hello", length=10)
    def test_lpad_with_empty_fill_text_violates_length(text, length):
        fill_text = ""
        result = _sqlite_lpad(text, length, fill_text)

        if result is not None:
>           assert len(result) == length, (
                f"lpad with empty fill_text should still return exact length. "
                f"Expected length {length}, got {len(result)}: {result!r}"
            )
E           AssertionError: lpad with empty fill_text should still return exact length. Expected length 10, got 5: 'hello'
E           assert 5 == 10
E            +  where 5 = len('hello')
E           Falsifying explicit example: test_lpad_with_empty_fill_text_violates_length(
E               text='hello',
E               length=10,
E           )

hypo.py:12: AssertionError
________________ test_rpad_with_empty_fill_text_violates_length ________________

    @given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
>   @example(text="hello", length=10)
                   ^^^

hypo.py:19:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

text = 'hello', length = 10

    @given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
    @example(text="hello", length=10)
    def test_rpad_with_empty_fill_text_violates_length(text, length):
        fill_text = ""
        result = _sqlite_rpad(text, length, fill_text)

        if result is not None:
>           assert len(result) == length, (
                f"rpad with empty fill_text should still return exact length. "
                f"Expected length {length}, got {len(result)}: {result!r}"
            )
E           AssertionError: rpad with empty fill_text should still return exact length. Expected length 10, got 5: 'hello'
E           assert 5 == 10
E            +  where 5 = len('hello')
E           Falsifying explicit example: test_rpad_with_empty_fill_text_violates_length(
E               text='hello',
E               length=10,
E           )

hypo.py:25: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_lpad_with_empty_fill_text_violates_length - AssertionErr...
FAILED hypo.py::test_rpad_with_empty_fill_text_violates_length - AssertionErr...
============================== 2 failed in 0.70s ===============================
```
</details>

## Reproducing the Bug

```python
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

# Test case from the bug report
text = "hello"
length = 10
fill_text = ""

print("Testing LPAD and RPAD with empty fill_text")
print("=" * 50)
print(f"Input: text='{text}', length={length}, fill_text='{fill_text}'")
print()

# Test LPAD
result_lpad = _sqlite_lpad(text, length, fill_text)
print(f"LPAD result: {result_lpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_lpad)}")
print()

# Test RPAD
result_rpad = _sqlite_rpad(text, length, fill_text)
print(f"RPAD result: {result_rpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_rpad)}")
print()

# Test edge cases
print("Additional edge cases:")
print("-" * 30)

# Case 1: Text longer than target length with empty padding
longer_text = "verylongtext"
target_length = 5
result_lpad_truncate = _sqlite_lpad(longer_text, target_length, fill_text)
result_rpad_truncate = _sqlite_rpad(longer_text, target_length, fill_text)
print(f"Text longer than target ('{longer_text}', length={target_length}):")
print(f"  LPAD: {result_lpad_truncate!r} (length: {len(result_lpad_truncate)})")
print(f"  RPAD: {result_rpad_truncate!r} (length: {len(result_rpad_truncate)})")
print()

# Case 2: Normal case with non-empty padding (for comparison)
normal_fill = "x"
result_lpad_normal = _sqlite_lpad(text, length, normal_fill)
result_rpad_normal = _sqlite_rpad(text, length, normal_fill)
print(f"Normal case with fill_text='{normal_fill}':")
print(f"  LPAD: {result_lpad_normal!r} (length: {len(result_lpad_normal)})")
print(f"  RPAD: {result_rpad_normal!r} (length: {len(result_rpad_normal)})")
print()

# Case 3: NULL/None handling
result_lpad_none = _sqlite_lpad(text, length, None)
result_rpad_none = _sqlite_rpad(text, length, None)
print(f"With fill_text=None:")
print(f"  LPAD: {result_lpad_none!r}")
print(f"  RPAD: {result_rpad_none!r}")
```

<details>

<summary>
LPAD/RPAD return incorrect length when fill_text is empty
</summary>
```
Testing LPAD and RPAD with empty fill_text
==================================================
Input: text='hello', length=10, fill_text=''

LPAD result: 'hello'
Expected length: 10, Actual length: 5

RPAD result: 'hello'
Expected length: 10, Actual length: 5

Additional edge cases:
------------------------------
Text longer than target ('verylongtext', length=5):
  LPAD: 'veryl' (length: 5)
  RPAD: 'veryl' (length: 5)

Normal case with fill_text='x':
  LPAD: 'xxxxxhello' (length: 10)
  RPAD: 'helloxxxxx' (length: 10)

With fill_text=None:
  LPAD: None
  RPAD: None
```
</details>

## Why This Is A Bug

This violates the fundamental contract of SQL LPAD/RPAD functions in multiple critical ways:

1. **Violation of Length Invariant**: SQL LPAD and RPAD functions must ALWAYS return either:
   - A string of EXACTLY the specified length, OR
   - NULL if padding to the required length is impossible

   Django's implementation returns the original string unchanged when `fill_text` is empty and `len(text) < length`, producing a string of incorrect length. This breaks any code that depends on fixed-width formatting.

2. **Deviation from SQL Standards**: All major SQL databases handle empty padding strings consistently:
   - **MySQL**: Returns NULL when padding string is empty (`SELECT LPAD('ABCD', 22, '')` → NULL)
   - **Oracle**: Returns NULL (empty string is equivalent to NULL in Oracle)
   - **PostgreSQL**: Returns NULL for empty padding strings

   Django aims to provide consistent database functionality across backends. The SQLite backend should emulate standard SQL behavior.

3. **Data Integrity Risk**: Applications using LPAD/RPAD expect predictable output lengths for:
   - Fixed-width file generation (e.g., legacy system interfaces)
   - Database constraints expecting specific column widths
   - String formatting in reports and data exports
   - Text alignment in terminal or fixed-width displays

4. **Silent Failure**: The current implementation silently returns incorrect results rather than signaling an error condition (returning NULL), making bugs harder to detect in production.

## Relevant Context

The bug exists in Django's SQLite3 backend implementation at `/django/db/backends/sqlite3/_functions.py`:

- **`_sqlite_lpad` function** (lines 389-395): When `fill_text=""`, the expression `(fill_text * length)[:delta]` produces an empty string, causing `"" + text` to return the original text unchanged.

- **`_sqlite_rpad` function** (lines 438-441): Similarly, `(text + fill_text * length)[:length]` with empty `fill_text` just returns the original text when it's shorter than the target length.

The functions correctly handle NULL values (returning NULL when any parameter is NULL) but fail to check for empty strings, which require the same treatment according to SQL standards.

Django documentation: https://docs.djangoproject.com/en/stable/ref/models/database-functions/#lpad
Source code: https://github.com/django/django/blob/main/django/db/backends/sqlite3/_functions.py

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -389,6 +389,9 @@ def _sqlite_lpad(text, length, fill_text):
 def _sqlite_lpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    # Return NULL for empty padding string (matches MySQL/Oracle/PostgreSQL behavior)
+    if fill_text == "":
+        return None
     delta = length - len(text)
     if delta <= 0:
         return text[:length]
@@ -438,6 +441,9 @@ def _sqlite_rpad(text, length, fill_text):
 def _sqlite_rpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    # Return NULL for empty padding string (matches MySQL/Oracle/PostgreSQL behavior)
+    if fill_text == "":
+        return None
     return (text + fill_text * length)[:length]
```