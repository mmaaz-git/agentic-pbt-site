# Bug Report: Django SQLite LPAD/RPAD Negative Length Handling

**Target**: `django.db.backends.sqlite3._functions._sqlite_lpad` and `django.db.backends.sqlite3._functions._sqlite_rpad`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_sqlite_lpad` and `_sqlite_rpad` functions incorrectly handle negative length values, returning truncated strings instead of empty strings as per SQL standard behavior (PostgreSQL, MySQL).

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
```

**Failing input**: `LPAD('00', -1, '0')` and `RPAD('00', -1, '0')`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

result = _sqlite_lpad('00', -1, '0')
print(result)

result = _sqlite_rpad('hello', -2, 'X')
print(result)
```

Expected output:
```
''
''
```

Actual output:
```
'0'
'hel'
```

## Why This Is A Bug

According to SQL standards implemented in major databases:

**PostgreSQL/Redshift**: "If length is zero or a negative number, the result of the function is an empty string"

**MySQL/MariaDB**: Similar behavior - returns empty string for negative lengths

The current Django implementation uses Python's negative slicing (`text[:length]`), which when `length` is negative, returns all characters except the last `abs(length)` characters. For example:
- `'hello'[:-2]` returns `'hel'` (all but last 2 chars)
- `'00'[:-1]` returns `'0'` (all but last char)

This violates the expected SQL behavior where negative lengths should return empty strings.

## Fix

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
@@ -438,4 +440,6 @@ def _sqlite_rpad(text, length, fill_text):
 def _sqlite_rpad(text, length, fill_text):
     if text is None or length is None or fill_text is None:
         return None
+    if length <= 0:
+        return ""
     return (text + fill_text * length)[:length]
```