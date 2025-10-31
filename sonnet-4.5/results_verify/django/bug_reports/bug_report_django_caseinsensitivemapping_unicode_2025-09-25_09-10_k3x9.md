# Bug Report: CaseInsensitiveMapping Unicode Case Folding

**Target**: `django.utils.datastructures.CaseInsensitiveMapping`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`CaseInsensitiveMapping` fails to provide case-insensitive access for Unicode characters where `str.upper()` and `str.lower()` don't round-trip, such as 'µ' (MICRO SIGN U+00B5).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.utils.datastructures import CaseInsensitiveMapping

@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_caseinsensitivemapping_case_insensitive(d):
    assume(len(d) > 0)
    mapping = CaseInsensitiveMapping(d)

    for key in d:
        assert mapping[key] == d[key]
        assert mapping[key.upper()] == d[key]
        assert mapping[key.lower()] == d[key]
```

**Failing input**: `{'µ': 0}`

## Reproducing the Bug

```python
from django.utils.datastructures import CaseInsensitiveMapping

mapping = CaseInsensitiveMapping({'µ': 0})

print(mapping['µ'])

print(mapping['Μ'])
```

Output:
```
0
KeyError: 'μ'
```

The issue occurs because:
- `'µ'` (MICRO SIGN U+00B5) lowercases to itself `'µ'`
- `'µ'.upper()` becomes `'Μ'` (GREEK CAPITAL LETTER MU U+039C)
- `'Μ'.lower()` becomes `'μ'` (GREEK SMALL LETTER MU U+03BC)
- So `'µ'.lower()` ≠ `'µ'.upper().lower()`

## Why This Is A Bug

The class documentation states it provides "case-insensitive key lookups", but it fails for certain Unicode characters. The implementation assumes `key.lower() == key.upper().lower()` which is not always true in Unicode.

This violates the expected invariant: if a key exists in the mapping, accessing it with any case variation should work.

## Fix

Use Unicode case folding (`str.casefold()`) instead of `str.lower()` for proper case-insensitive comparison:

```diff
--- a/django/utils/datastructures.py
+++ b/django/utils/datastructures.py
@@ -302,10 +302,10 @@ class CaseInsensitiveMapping(Mapping):
     """

     def __init__(self, data):
-        self._store = {k.lower(): (k, v) for k, v in self._unpack_items(data)}
+        self._store = {k.casefold(): (k, v) for k, v in self._unpack_items(data)}

     def __getitem__(self, key):
-        return self._store[key.lower()][1]
+        return self._store[key.casefold()][1]

     def __len__(self):
         return len(self._store)
@@ -313,7 +313,7 @@ class CaseInsensitiveMapping(Mapping):
     def __eq__(self, other):
         return isinstance(other, Mapping) and {
-            k.lower(): v for k, v in self.items()
-        } == {k.lower(): v for k, v in other.items()}
+            k.casefold(): v for k, v in self.items()
+        } == {k.casefold(): v for k, v in other.items()}

     def __iter__(self):
```