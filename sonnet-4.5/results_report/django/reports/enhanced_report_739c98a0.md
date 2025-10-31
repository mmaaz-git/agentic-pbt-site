# Bug Report: CaseInsensitiveMapping Unicode Case Folding Failure

**Target**: `django.utils.datastructures.CaseInsensitiveMapping`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CaseInsensitiveMapping fails to provide case-insensitive key lookups for Unicode characters where str.upper() and str.lower() transformations don't round-trip, causing KeyError when accessing keys with their uppercase forms.

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

<details>

<summary>
**Failing input**: `{'ß': 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 15, in <module>
    test_caseinsensitivemapping_case_insensitive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 5, in test_caseinsensitivemapping_case_insensitive
    def test_caseinsensitivemapping_case_insensitive(d):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 11, in test_caseinsensitivemapping_case_insensitive
    assert mapping[key.upper()] == d[key]
           ~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/utils/datastructures.py", line 308, in __getitem__
    return self._store[key.lower()][1]
           ~~~~~~~~~~~^^^^^^^^^^^^^
KeyError: 'ss'
Falsifying example: test_caseinsensitivemapping_case_insensitive(
    d={'ß': 0},
)
```
</details>

## Reproducing the Bug

```python
from django.utils.datastructures import CaseInsensitiveMapping

# Test case 1: German eszett (found by Hypothesis)
print("Test case 1: German eszett 'ß'")
mapping = CaseInsensitiveMapping({'ß': 0})
print(f"Access with 'ß': {mapping['ß']}")

try:
    print(f"'ß'.upper() = '{('ß'.upper())}'")
    print(f"Access with 'SS' (uppercase): {mapping['SS']}")
except KeyError as e:
    print(f"KeyError when accessing with uppercase: {e}")

print(f"'ß'.lower() = '{('ß'.lower())}' (stays as 'ß')")
print(f"'ß'.upper() = '{('ß'.upper())}' (becomes 'SS')")
print(f"'SS'.lower() = '{('SS'.lower())}' (becomes 'ss')")
print(f"Key stored internally: 'ß'.lower() = 'ß'")
print(f"Key looked up: 'SS'.lower() = 'ss' (mismatch!)")

print("\n" + "="*50 + "\n")

# Test case 2: Micro sign (from original report)
print("Test case 2: Micro sign 'µ'")
mapping2 = CaseInsensitiveMapping({'µ': 0})
print(f"Access with 'µ' (MICRO SIGN): {mapping2['µ']}")

try:
    print(f"'µ'.upper() = '{('µ'.upper())}'")
    print(f"Access with 'Μ' (uppercase): {mapping2['Μ']}")
except KeyError as e:
    print(f"KeyError when accessing with uppercase: {e}")

print(f"'µ' (MICRO SIGN) = U+{ord('µ'):04X}")
print(f"'µ'.lower() = '{('µ'.lower())}' = U+{ord('µ'.lower()):04X}")
print(f"'µ'.upper() = '{('µ'.upper())}' = U+{ord('µ'.upper()):04X}")
print(f"'µ'.upper().lower() = '{('µ'.upper().lower())}' = U+{ord('µ'.upper().lower()):04X}")
```

<details>

<summary>
KeyError when accessing with uppercase forms of 'ß' and 'µ'
</summary>
```
Test case 1: German eszett 'ß'
Access with 'ß': 0
'ß'.upper() = 'SS'
KeyError when accessing with uppercase: 'ss'
'ß'.lower() = 'ß' (stays as 'ß')
'ß'.upper() = 'SS' (becomes 'SS')
'SS'.lower() = 'ss' (becomes 'ss')
Key stored internally: 'ß'.lower() = 'ß'
Key looked up: 'SS'.lower() = 'ss' (mismatch!)

==================================================

Test case 2: Micro sign 'µ'
Access with 'µ' (MICRO SIGN): 0
'µ'.upper() = 'Μ'
KeyError when accessing with uppercase: 'μ'
'µ' (MICRO SIGN) = U+00B5
'µ'.lower() = 'µ' = U+00B5
'µ'.upper() = 'Μ' = U+039C
'µ'.upper().lower() = 'μ' = U+03BC
```
</details>

## Why This Is A Bug

The CaseInsensitiveMapping class documentation promises "case-insensitive key lookups" (line 288 of datastructures.py) and shows examples where accessing a key with any case variation should work. However, the implementation violates this contract for certain Unicode characters.

The bug occurs because the implementation assumes that `key.lower() == key.upper().lower()`, which is false for several Unicode characters:

1. **German eszett 'ß'**:
   - Stores as `'ß'.lower() = 'ß'`
   - But uppercase lookup tries to find `'SS'.lower() = 'ss'`
   - These don't match, causing KeyError

2. **Micro sign 'µ' (U+00B5)**:
   - Stores as `'µ'.lower() = 'µ'` (U+00B5)
   - But uppercase lookup tries to find `'Μ'.lower() = 'μ'` (U+03BC, different character!)
   - These are different Unicode characters, causing KeyError

This violates the fundamental invariant of case-insensitive access: if a key exists in the mapping, it should be accessible using any case variation of that key.

## Relevant Context

The issue affects real-world use cases:
- German text processing (ß is commonly used in German)
- Scientific/mathematical applications (µ for micro units)
- Turkish text (dotted/dotless i has similar issues)
- Any internationalized application handling non-ASCII text

The Python documentation explicitly recommends using `str.casefold()` for case-insensitive string matching to handle these Unicode edge cases properly. See: https://docs.python.org/3/library/stdtypes.html#str.casefold

The current implementation at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/utils/datastructures.py` lines 305-316 uses `str.lower()` which is insufficient for proper Unicode case-insensitive comparison.

## Proposed Fix

Replace `str.lower()` with `str.casefold()` for proper Unicode case-insensitive comparison:

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