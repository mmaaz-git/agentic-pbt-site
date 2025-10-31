# Bug Report: Cython.Plex.Regexps.RE.wrong_type Python 3 Incompatibility

**Target**: `Cython.Plex.Regexps.RE.wrong_type`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RE.wrong_type` method uses `types.InstanceType` which was removed in Python 3, causing an AttributeError when RE validation fails instead of raising the intended PlexTypeError.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex import Seq, Str
from Cython.Plex.Errors import PlexTypeError

@given(st.text(alphabet='abc', min_size=1, max_size=5))
@settings(max_examples=200)
def test_seq_rejects_non_re_args(s):
    with pytest.raises(PlexTypeError):
        Seq(Str(s), "not an RE")
```

**Failing input**: `s='a'` (or any string)

## Reproducing the Bug

```python
from Cython.Plex import Seq, Str

try:
    seq = Seq(Str('a'), "not an RE")
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Expected: PlexTypeError")
```

Output:
```
AttributeError: module 'types' has no attribute 'InstanceType'
Expected: PlexTypeError
```

## Why This Is A Bug

The `wrong_type` method at line 167 in Regexps.py checks `if type(value) == types.InstanceType:` but `types.InstanceType` was removed in Python 3.0 (it was specific to old-style classes in Python 2). This causes an AttributeError whenever the validation code tries to report a type error, completely breaking the error handling for invalid RE arguments.

This is a high-severity bug because:
1. It affects basic API validation for all RE constructors (Seq, Alt, Rep1, etc.)
2. Users get cryptic AttributeError instead of helpful PlexTypeError messages
3. The bug occurs on any invalid input, making it easy to trigger

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -164,7 +164,7 @@ class RE:
                                             num, self.__class__.__name__, repr(value)))

     def wrong_type(self, num, value, expected):
-        if type(value) == types.InstanceType:
+        if hasattr(value, '__class__') and hasattr(value.__class__, '__module__'):
             got = "%s.%s instance" % (
                 value.__class__.__module__, value.__class__.__name__)
         else:
```