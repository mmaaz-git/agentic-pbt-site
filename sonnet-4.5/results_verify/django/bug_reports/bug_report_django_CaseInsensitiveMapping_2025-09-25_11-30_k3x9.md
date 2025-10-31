# Bug Report: CaseInsensitiveMapping Case-Folding Asymmetry

**Target**: `django.utils.datastructures.CaseInsensitiveMapping`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CaseInsensitiveMapping fails to retrieve values when the lookup key's uppercase and lowercase transformations are not symmetric, such as with the German letter 'ß' (sharp s), which uppercases to 'SS'.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.utils.datastructures import CaseInsensitiveMapping


@given(st.dictionaries(st.text(), st.text()))
@settings(max_examples=500)
def test_case_insensitive_mapping_access(d):
    cim = CaseInsensitiveMapping(d)
    for key, value in d.items():
        assert cim.get(key) == value
        assert cim.get(key.upper()) == value
        assert cim.get(key.lower()) == value
```

**Failing input**: `d={'ß': ''}`

## Reproducing the Bug

```python
from django.utils.datastructures import CaseInsensitiveMapping

cim = CaseInsensitiveMapping({'ß': 'value'})

print(cim.get('ß'))
print(cim.get('SS'))

cim['SS']
```

**Output:**
```
value
None
KeyError: 'ss'
```

## Why This Is A Bug

The CaseInsensitiveMapping documentation states it provides "case-insensitive key lookups." Users would reasonably expect that if they can store a value with key 'ß', they should be able to retrieve it using any case variation, including the uppercase form 'SS'.

The bug occurs because:
1. When 'ß' is stored, it's normalized to 'ß' (since `'ß'.lower()` == 'ß')
2. When looking up 'SS', it's normalized to 'ss' (since `'SS'.lower()` == 'ss')
3. 'ss' ≠ 'ß', so the lookup fails

This breaks the case-insensitive lookup contract for Unicode characters where `str.upper().lower() != str.lower()`.

Other affected characters include:
- Turkish 'İ' (capital I with dot): `'İ'.lower()` → 'i̇', but `'I'.lower()` → 'i'
- Greek 'Σ' (capital sigma): different lowercase forms depending on position

## Fix

Use case-folding (`str.casefold()`) instead of lowercasing for normalization. Case-folding is specifically designed for caseless matching and handles these Unicode edge cases correctly.

```diff
--- a/django/utils/datastructures.py
+++ b/django/utils/datastructures.py
@@ -412,11 +412,11 @@ class CaseInsensitiveMapping(Mapping):
     """

     def __init__(self, data):
-        self._store = {k.lower(): (k, v) for k, v in self._unpack_items(data)}
+        self._store = {k.casefold(): (k, v) for k, v in self._unpack_items(data)}

     def __getitem__(self, key):
-        return self._store[key.lower()][1]
+        return self._store[key.casefold()][1]

     def __len__(self):
         return len(self._store)
@@ -424,7 +424,7 @@ class CaseInsensitiveMapping(Mapping):
     def __eq__(self, other):
         return isinstance(other, Mapping) and {
-            k.lower(): v for k, v in self.items()
-        } == {k.lower(): v for k, v in other.items()}
+            k.casefold(): v for k, v in self.items()
+        } == {k.casefold(): v for k, v in other.items()}
```