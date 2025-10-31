# Bug Report: scipy.constants.find() Incomplete Search

**Target**: `scipy.constants.find()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `find()` function in `scipy.constants` only searches through current CODATA 2022 constants, missing 90 obsolete constants from earlier CODATA versions that are still accessible via `physical_constants` dictionary. This violates the function's documented behavior which states "By default, return all keys."

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st
import scipy.constants as const
from scipy.constants import _codata


@given(st.sampled_from(list(const.physical_constants.keys())))
def test_find_searches_all_physical_constants(key):
    """All keys in physical_constants should be findable."""
    search_term = key.split()[0][:4]
    find_result = const.find(search_term)
    expected_matches = [k for k in const.physical_constants.keys()
                       if search_term.lower() in k.lower()]
    if key in expected_matches:
        assert key in find_result
```

**Failing input**: Any obsolete constant key, e.g., `'muon Compton wavelength over 2 pi'`

## Reproducing the Bug

```python
import scipy.constants as const

all_keys = const.physical_constants.keys()
find_result = const.find(None)

print(f"physical_constants has {len(list(all_keys))} keys")
print(f"find(None) returns {len(find_result)} keys")
print(f"Missing: {len(list(all_keys)) - len(find_result)} keys")

assert len(list(all_keys)) == len(find_result)
```

Output:
```
physical_constants has 448 keys
find(None) returns 358 keys
Missing: 90 keys
AssertionError
```

## Why This Is A Bug

1. **Documentation violation**: The docstring explicitly states "Sub-string to search keys for. By default, return all keys." But `find(None)` only returns 358 out of 448 keys.

2. **Inconsistent API**: Users can access obsolete constants via `physical_constants['muon Compton wavelength over 2 pi']` or `value('muon Compton wavelength over 2 pi')` (with a warning), but `find()` cannot discover them.

3. **Broken search**: When searching for a substring like `'muon'`, users expect to find ALL matching keys including obsolete ones, but the function only searches current CODATA 2022 constants.

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