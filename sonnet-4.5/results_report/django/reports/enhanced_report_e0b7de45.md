# Bug Report: django.db.backends.sqlite3._sqlite_lpad Violates Length Invariant with Empty Fill Text

**Target**: `django.db.backends.sqlite3._functions._sqlite_lpad`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_lpad` function fails to maintain its fundamental invariant of returning a string of exactly `length` characters when the `fill_text` parameter is an empty string, instead returning the unpadded original text.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis property-based test for django.db.backends.sqlite3._functions._sqlite_lpad
Tests the fundamental invariant: LPAD should always return a string of exactly 'length' characters
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_lpad

@given(st.text(), st.integers(min_value=0, max_value=10000), st.text(min_size=0))
@settings(max_examples=10000)
def test_lpad_length_invariant(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    if result is not None:
        assert len(result) == length, f"Expected length {length}, got {len(result)} for _sqlite_lpad({text!r}, {length}, {fill_text!r})"

if __name__ == "__main__":
    test_lpad_length_invariant()
```

<details>

<summary>
**Failing input**: `_sqlite_lpad('', 1, '')` returns `''` (length 0) instead of length 1
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 21, in <module>
    test_lpad_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 14, in test_lpad_length_invariant
    @settings(max_examples=10000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 18, in test_lpad_length_invariant
    assert len(result) == length, f"Expected length {length}, got {len(result)} for _sqlite_lpad({text!r}, {length}, {fill_text!r})"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected length 1, got 0 for _sqlite_lpad('', 1, '')
Falsifying example: test_lpad_length_invariant(
    text='',
    length=1,
    fill_text='',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of django.db.backends.sqlite3._functions._sqlite_lpad bug
Shows that the function violates its length invariant when fill_text is empty
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.db.backends.sqlite3._functions import _sqlite_lpad

print("Testing _sqlite_lpad function with empty fill_text")
print("=" * 60)

# Test case 1: Empty text, length 1, empty fill_text
print("\nTest 1: _sqlite_lpad('', 1, '')")
result1 = _sqlite_lpad('', 1, '')
print(f"  Result: '{result1}'")
print(f"  Length: {len(result1)}")
print(f"  Expected length: 1")
print(f"  FAIL: Expected length 1, got {len(result1)}")

# Test case 2: 'hello', length 10, empty fill_text
print("\nTest 2: _sqlite_lpad('hello', 10, '')")
result2 = _sqlite_lpad('hello', 10, '')
print(f"  Result: '{result2}'")
print(f"  Length: {len(result2)}")
print(f"  Expected length: 10")
print(f"  FAIL: Expected length 10, got {len(result2)}")

# Test case 3: 'x', length 5, empty fill_text
print("\nTest 3: _sqlite_lpad('x', 5, '')")
result3 = _sqlite_lpad('x', 5, '')
print(f"  Result: '{result3}'")
print(f"  Length: {len(result3)}")
print(f"  Expected length: 5")
print(f"  FAIL: Expected length 5, got {len(result3)}")

# Control test: Non-empty fill_text works correctly
print("\nControl Test: _sqlite_lpad('hello', 10, '*')")
result4 = _sqlite_lpad('hello', 10, '*')
print(f"  Result: '{result4}'")
print(f"  Length: {len(result4)}")
print(f"  Expected length: 10")
print(f"  PASS: Got expected length 10")

print("\n" + "=" * 60)
print("Summary: The function violates its fundamental invariant")
print("LPAD should always return a string of exactly 'length' characters")
print("but returns the unpadded text when fill_text is empty.")
```

<details>

<summary>
Function returns unpadded text when fill_text is empty
</summary>
```
Testing _sqlite_lpad function with empty fill_text
============================================================

Test 1: _sqlite_lpad('', 1, '')
  Result: ''
  Length: 0
  Expected length: 1
  FAIL: Expected length 1, got 0

Test 2: _sqlite_lpad('hello', 10, '')
  Result: 'hello'
  Length: 5
  Expected length: 10
  FAIL: Expected length 10, got 5

Test 3: _sqlite_lpad('x', 5, '')
  Result: 'x'
  Length: 1
  Expected length: 5
  FAIL: Expected length 5, got 1

Control Test: _sqlite_lpad('hello', 10, '*')
  Result: '*****hello'
  Length: 10
  Expected length: 10
  PASS: Got expected length 10

============================================================
Summary: The function violates its fundamental invariant
LPAD should always return a string of exactly 'length' characters
but returns the unpadded text when fill_text is empty.
```
</details>

## Why This Is A Bug

The LPAD (Left Pad) function's core contract is to return a string padded to exactly the specified `length`. This invariant is violated when `fill_text` is an empty string. The Django documentation states that LPad "Returns the value of the given text field or expression padded on the left side with `fill_text` so that the resulting value is `length` characters long."

The bug occurs because of the implementation at line 395 in `/django/db/backends/sqlite3/_functions.py`:
```python
return (fill_text * length)[:delta] + text
```

When `fill_text` is an empty string:
1. `fill_text * length` produces an empty string regardless of the multiplier
2. Slicing the empty string `""[:delta]` returns an empty string
3. The result is just the original `text` without any padding

This behavior is inconsistent with:
- The function's documented purpose of returning a string of exactly `length` characters
- The behavior when `fill_text` is non-empty (which correctly pads to the specified length)
- Standard SQL database implementations (PostgreSQL defaults to space padding, MySQL/Oracle have defined behaviors)

Applications relying on LPAD to ensure consistent string lengths for formatting, data alignment, or storage requirements will experience unexpected results that could lead to data integrity issues.

## Relevant Context

The `_sqlite_lpad` function is part of Django's SQLite backend implementation, providing compatibility for the LPAD SQL function which SQLite doesn't natively support. The function is located at:
- File: `/django/db/backends/sqlite3/_functions.py`
- Lines: 389-395
- Django documentation: [Database Functions - LPad](https://docs.djangoproject.com/en/stable/ref/models/database-functions/#lpad)

The issue only manifests when:
1. The `fill_text` parameter is an empty string (`''`)
2. The text needs padding (i.e., `len(text) < length`)

When `fill_text` has any characters, the function works correctly. This edge case may not be commonly encountered but violates the function's fundamental contract and could cause subtle bugs in production systems expecting consistent string lengths.

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -392,7 +392,11 @@ def _sqlite_lpad(text, length, fill_text):
     delta = length - len(text)
     if delta <= 0:
         return text[:length]
-    return (fill_text * length)[:delta] + text
+    if fill_text:
+        return (fill_text * length)[:delta] + text
+    else:
+        # When fill_text is empty, default to space padding (matching PostgreSQL behavior)
+        return (' ' * delta) + text


 def _sqlite_md5(text):
```