# Bug Report: django.utils.datastructures.CaseInsensitiveMapping Fails with Unicode Characters Having Asymmetric Case Transformations

**Target**: `django.utils.datastructures.CaseInsensitiveMapping`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CaseInsensitiveMapping fails to provide case-insensitive key lookups for Unicode characters where uppercase and lowercase transformations are not symmetric, such as the German 'ß' (eszett) which uppercases to 'SS' but remains 'ß' when lowercased.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for CaseInsensitiveMapping."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from django.utils.datastructures import CaseInsensitiveMapping


@given(st.dictionaries(st.text(), st.text()))
@settings(max_examples=500)
@example({'ß': ''})  # Add the German eszett as a specific example to test
def test_case_insensitive_mapping_access(d):
    cim = CaseInsensitiveMapping(d)
    for key, value in d.items():
        assert cim.get(key) == value
        assert cim.get(key.upper()) == value
        assert cim.get(key.lower()) == value


if __name__ == "__main__":
    # Run the test
    test_case_insensitive_mapping_access()
```

<details>

<summary>
**Failing input**: `d={'ß': ''}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 24, in <module>
    test_case_insensitive_mapping_access()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 12, in test_case_insensitive_mapping_access
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 18, in test_case_insensitive_mapping_access
    assert cim.get(key.upper()) == value
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying explicit example: test_case_insensitive_mapping_access(
    d={'ß': ''},
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for CaseInsensitiveMapping bug with German ß character."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.datastructures import CaseInsensitiveMapping

# Create a CaseInsensitiveMapping with German eszett character
cim = CaseInsensitiveMapping({'ß': 'value'})

# Test retrieving with the original key
print("cim.get('ß'):", cim.get('ß'))

# Test retrieving with uppercase version (SS)
print("cim.get('SS'):", cim.get('SS'))

# Try direct access with uppercase (this will raise an error)
try:
    result = cim['SS']
    print("cim['SS']:", result)
except KeyError as e:
    print(f"cim['SS'] raised KeyError: {e}")

# Show the case transformation issue
print("\nCase transformations:")
print(f"'ß'.upper() = '{('ß').upper()}'")
print(f"'ß'.lower() = '{('ß').lower()}'")
print(f"'SS'.lower() = '{('SS').lower()}'")
print(f"'ss'.lower() = '{('ss').lower()}'")

print("\nUsing casefold() (proper Unicode case-insensitive matching):")
print(f"'ß'.casefold() = '{('ß').casefold()}'")
print(f"'SS'.casefold() = '{('SS').casefold()}'")
print(f"'ss'.casefold() = '{('ss').casefold()}'")
```

<details>

<summary>
KeyError when accessing with uppercase form 'SS'
</summary>
```
cim.get('ß'): value
cim.get('SS'): None
cim['SS'] raised KeyError: 'ss'

Case transformations:
'ß'.upper() = 'SS'
'ß'.lower() = 'ß'
'SS'.lower() = 'ss'
'ss'.lower() = 'ss'

Using casefold() (proper Unicode case-insensitive matching):
'ß'.casefold() = 'ss'
'SS'.casefold() = 'ss'
'ss'.casefold() = 'ss'
```
</details>

## Why This Is A Bug

The CaseInsensitiveMapping class documentation states it provides "case-insensitive key lookups" without any qualification or limitation. This creates a reasonable expectation that if a value can be stored with a key in one case, it should be retrievable using any case variation of that key.

The bug specifically violates this contract because:

1. **Asymmetric case transformations break lookups**: When storing 'ß', the implementation normalizes it using `.lower()` which produces 'ß' (unchanged). When looking up 'SS' (the uppercase form of 'ß'), it normalizes to 'ss', which doesn't match the stored key 'ß'.

2. **The documentation makes no mention of Unicode limitations**: The class accepts Unicode strings and its documentation example only shows ASCII characters, but doesn't warn about Unicode edge cases.

3. **Python provides the correct tool**: The `.casefold()` method exists specifically for case-insensitive Unicode string matching and would handle this correctly ('ß'.casefold() == 'SS'.casefold() == 'ss'.casefold() == 'ss').

4. **Real-world impact**: This affects commonly used characters in major languages:
   - German 'ß' (eszett/sharp s) - used in one of Europe's most spoken languages
   - Greek 'µ' (micro sign) which uppercases to 'Μ' (capital mu) but they lowercase differently
   - Turkish dotted/dotless i variations
   - Various other Unicode characters with special casing rules

## Relevant Context

The CaseInsensitiveMapping class is located in `/django/utils/datastructures.py` (lines 286-345) and is primarily used by Django for HTTP header handling through the `HttpHeaders` and `ResponseHeaders` classes. While HTTP headers are typically ASCII, the class itself accepts any string keys and makes no restrictions on Unicode usage.

The current implementation uses Python's `str.lower()` method for normalization:
- Line 305: `self._store = {k.lower(): (k, v) for k, v in self._unpack_items(data)}`
- Line 308: `return self._store[key.lower()][1]`
- Lines 314-316: Equality comparison also uses `.lower()`

According to the [Python documentation on casefold()](https://docs.python.org/3/library/stdtypes.html#str.casefold), the casefold() method is "similar to lowercasing but more aggressive because it is intended to remove all case distinctions in a string" and is "suitable for caseless matching."

The [Unicode Standard](https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf#G33992) defines case folding as the process for case-insensitive comparison, which Python's `.casefold()` implements correctly.

## Proposed Fix

Replace `.lower()` with `.casefold()` for proper Unicode case-insensitive matching:

```diff
--- a/django/utils/datastructures.py
+++ b/django/utils/datastructures.py
@@ -302,11 +302,11 @@ class CaseInsensitiveMapping(Mapping):
     """

     def __init__(self, data):
-        self._store = {k.lower(): (k, v) for k, v in self._unpack_items(data)}
+        self._store = {k.casefold(): (k, v) for k, v in self._unpack_items(data)}

     def __getitem__(self, key):
-        return self._store[key.lower()][1]
+        return self._store[key.casefold()][1]

     def __len__(self):
         return len(self._store)
@@ -311,8 +311,8 @@ class CaseInsensitiveMapping(Mapping):

     def __eq__(self, other):
         return isinstance(other, Mapping) and {
-            k.lower(): v for k, v in self.items()
-        } == {k.lower(): v for k, v in other.items()}
+            k.casefold(): v for k, v in self.items()
+        } == {k.casefold(): v for k, v in other.items()}

     def __iter__(self):
         return (original_key for original_key, value in self._store.values())
```