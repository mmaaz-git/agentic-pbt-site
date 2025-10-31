# Bug Report: scipy.constants.find() Missing 90 Physical Constants

**Target**: `scipy.constants.find`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find()` function only searches through CODATA 2022 constants (355 keys) instead of all available physical constants (445 keys), making 90 deprecated constants unsearchable despite being present in `physical_constants`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.constants import find, physical_constants

@given(sub=st.text(min_size=1, max_size=20))
def test_find_substring_match(sub):
    result = find(sub, disp=False)
    for key in result:
        assert sub.lower() in key.lower()

def test_find_none_returns_all():
    result_none = find(None, disp=False)
    all_keys = sorted(physical_constants.keys())
    assert result_none == all_keys
```

**Failing input**: `None` (or any deprecated constant name like `"Bohr magneton in inverse meters per tesla"`)

## Reproducing the Bug

```python
from scipy.constants import find, physical_constants

all_keys_via_find = find(None)
all_keys_from_dict = list(physical_constants.keys())

print(f"Keys via find(None): {len(all_keys_via_find)}")
print(f"Keys in physical_constants: {len(all_keys_from_dict)}")
print(f"Missing: {len(all_keys_from_dict) - len(all_keys_via_find)} keys")

missing = set(all_keys_from_dict) - set(all_keys_via_find)
key = sorted(missing)[0]
print(f"\nExample: '{key}'")
print(f"Exists in physical_constants: {key in physical_constants}")
print(f"Can find() locate it: {find(key) != []}")
```

Output:
```
Keys via find(None): 355
Keys in physical_constants: 445
Missing: 90 keys

Example: 'Bohr magneton in inverse meters per tesla'
Exists in physical_constants: True
Can find() locate it: False
```

## Why This Is A Bug

1. **Docstring promises**: The docstring states "By default, return all keys" when `sub=None`, but it only returns 355 out of 445 keys.

2. **Examples contradictory**: The docstring examples reference `physical_constants`, implying that `find()` searches that dictionary, but the implementation searches `_current_constants`.

3. **Inconsistent API**: Users can access deprecated constants via `physical_constants["deprecated key"]` but cannot discover them via `find()`.

4. **Real-world impact**: Scientists using older CODATA values for backward compatibility cannot discover which constants are available.

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -2248,9 +2248,9 @@ def find(sub: str | None = None, disp: bool = False) -> Any:

     """
     if sub is None:
-        result = list(_current_constants.keys())
+        result = list(physical_constants.keys())
     else:
-        result = [key for key in _current_constants
+        result = [key for key in physical_constants
                   if sub.lower() in key.lower()]

     result.sort()
```