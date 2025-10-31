# Bug Report: Cython.Plex.Regexps.RawCodeRange.calc_str AttributeError

**Target**: `Cython.Plex.Regexps.RawCodeRange.calc_str`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RawCodeRange.calc_str` method references non-existent attributes `self.code1` and `self.code2`, causing an AttributeError when the string representation is requested.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pytest
from Cython.Plex.Regexps import RawCodeRange

@given(st.integers(min_value=0, max_value=200),
       st.integers(min_value=0, max_value=200))
@settings(max_examples=300)
def test_rawcoderange_str_method(code1, code2):
    if code1 >= code2:
        return

    rcr = RawCodeRange(code1, code2)

    try:
        str_repr = str(rcr)
        assert str_repr is not None
    except AttributeError as e:
        if 'code1' in str(e) or 'code2' in str(e):
            pytest.fail(f"RawCodeRange.calc_str references non-existent attributes: {e}")
```

**Failing input**: `code1=0, code2=1` (or any valid code range)

## Reproducing the Bug

```python
from Cython.Plex.Regexps import RawCodeRange

rcr = RawCodeRange(50, 60)

try:
    s = str(rcr)
except AttributeError as e:
    print(f"AttributeError: {e}")
    print(f"rcr.range exists: {hasattr(rcr, 'range')}")
    print(f"rcr.code1 exists: {hasattr(rcr, 'code1')}")
    print(f"rcr.code2 exists: {hasattr(rcr, 'code2')}")
```

Output:
```
AttributeError: 'RawCodeRange' object has no attribute 'code1'
rcr.range exists: True
rcr.code1 exists: False
rcr.code2 exists: False
```

## Why This Is A Bug

The `RawCodeRange.__init__` method (lines 208-211) stores the code range as `self.range = (code1, code2)`, but the `calc_str` method (line 224) tries to access `self.code1` and `self.code2` which are never set. This causes an AttributeError whenever the string representation is needed (for debugging, logging, or error messages).

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -221,7 +221,7 @@ class RawCodeRange(RE):
                 initial_state.add_transition(self.lowercase_range, final_state)

     def calc_str(self):
-        return "CodeRange(%d,%d)" % (self.code1, self.code2)
+        return "CodeRange(%d,%d)" % self.range
```