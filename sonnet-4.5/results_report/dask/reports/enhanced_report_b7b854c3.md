# Bug Report: Django SQLite LPAD/RPAD Functions Incorrectly Handle Negative Length Values

**Target**: `django.db.backends.sqlite3._functions._sqlite_lpad` and `django.db.backends.sqlite3._functions._sqlite_rpad`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Django SQLite implementations of LPAD and RPAD functions incorrectly handle negative length values by returning truncated strings (via Python's negative slicing) instead of empty strings as per SQL standard behavior.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad


@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_lpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    assert result == "", f"LPAD({text!r}, {length}, {fill_text!r}) should return empty string for negative length, got {result!r}"


@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_rpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_rpad(text, length, fill_text)
    assert result == "", f"RPAD({text!r}, {length}, {fill_text!r}) should return empty string for negative length, got {result!r}"


if __name__ == "__main__":
    print("Running property-based tests for LPAD/RPAD negative length handling...")
    print()

    print("Testing LPAD with negative lengths:")
    test_lpad_negative_length_returns_empty()

    print("Testing RPAD with negative lengths:")
    test_rpad_negative_length_returns_empty()
```

<details>

<summary>
**Failing input**: `LPAD('00', -1, '0')`
</summary>
```
Running property-based tests for LPAD/RPAD negative length handling...

Testing LPAD with negative lengths:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 27, in <module>
    test_lpad_negative_length_returns_empty()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 9, in test_lpad_negative_length_returns_empty
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 12, in test_lpad_negative_length_returns_empty
    assert result == "", f"LPAD({text!r}, {length}, {fill_text!r}) should return empty string for negative length, got {result!r}"
           ^^^^^^^^^^^^
AssertionError: LPAD('00', -1, '0') should return empty string for negative length, got '0'
Falsifying example: test_lpad_negative_length_returns_empty(
    text='00',
    length=-1,
    fill_text='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

print("Testing _sqlite_lpad with negative length:")
print(f"_sqlite_lpad('00', -1, '0') = {repr(_sqlite_lpad('00', -1, '0'))}")
print(f"Expected: ''")
print()

print("Testing _sqlite_rpad with negative length:")
print(f"_sqlite_rpad('hello', -2, 'X') = {repr(_sqlite_rpad('hello', -2, 'X'))}")
print(f"Expected: ''")
print()

print("Additional test cases:")
print(f"_sqlite_lpad('test', -3, '*') = {repr(_sqlite_lpad('test', -3, '*'))}")
print(f"_sqlite_rpad('example', -5, '#') = {repr(_sqlite_rpad('example', -5, '#'))}")
```

<details>

<summary>
Functions return truncated strings instead of empty strings for negative lengths
</summary>
```
Testing _sqlite_lpad with negative length:
_sqlite_lpad('00', -1, '0') = '0'
Expected: ''

Testing _sqlite_rpad with negative length:
_sqlite_rpad('hello', -2, 'X') = 'hel'
Expected: ''

Additional test cases:
_sqlite_lpad('test', -3, '*') = 't'
_sqlite_rpad('example', -5, '#') = 'ex'
```
</details>

## Why This Is A Bug

The current implementation violates expected SQL behavior for LPAD/RPAD functions when given negative length values. The bug occurs because both functions use Python's slice notation `text[:length]` without checking if length is negative. When length is negative, Python interprets this as "all characters except the last N", resulting in truncated strings rather than empty strings.

**Expected behavior according to SQL standards:**
- **PostgreSQL/Redshift**: "If length is zero or a negative number, the result of the function is an empty string"
- **MySQL/MariaDB**: Returns NULL for negative lengths (though some versions return empty string)

**Current Django behavior:**
- `text[:length]` with negative length removes the last `abs(length)` characters
- Example: `'hello'[:-2]` returns `'hel'` (removes last 2 characters)
- This is Python's negative slicing behavior, not SQL padding behavior

This inconsistency breaks Django's promise of database abstraction, as code using LPAD/RPAD will behave differently on SQLite compared to PostgreSQL or MySQL backends.

## Relevant Context

The bug exists in Django's custom SQLite implementation since SQLite doesn't have native LPAD/RPAD functions. The functions are defined in:
- `/django/db/backends/sqlite3/_functions.py`

The issue affects both `_sqlite_lpad` (lines 389-396) and `_sqlite_rpad` (lines 438-441) functions.

Django documentation doesn't explicitly specify the expected behavior for negative lengths, but following standard SQL semantics (especially PostgreSQL, which Django often uses as reference) would be the expected approach.

This bug could affect applications that:
- Dynamically calculate padding lengths that might go negative
- Port code from PostgreSQL/MySQL to SQLite
- Rely on consistent behavior across database backends

## Proposed Fix

```diff
--- a/django/db/backends/sqlite3/_functions.py
+++ b/django/db/backends/sqlite3/_functions.py
@@ -389,6 +389,8 @@ def _sqlite_lpad(text, length, fill_text):
 def _sqlite_lpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    if length <= 0:
+        return ""
     delta = length - len(text)
     if delta <= 0:
         return text[:length]
@@ -438,6 +440,8 @@ def _sqlite_rpad(text, length, fill_text):
 def _sqlite_rpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    if length <= 0:
+        return ""
     return (text + fill_text * length)[:length]
```