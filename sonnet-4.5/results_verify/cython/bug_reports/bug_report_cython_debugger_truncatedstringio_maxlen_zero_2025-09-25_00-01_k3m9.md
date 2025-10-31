# Bug Report: Cython.Debugger.libpython.TruncatedStringIO maxlen=0 Ignored

**Target**: `Cython.Debugger.libpython.TruncatedStringIO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `TruncatedStringIO` class does not enforce its maximum length constraint when `maxlen=0` due to a falsy check, allowing unlimited data to be written despite the explicit zero-length limit.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Debugger.libpython import TruncatedStringIO, StringTruncated

@given(st.text(min_size=1))
def test_maxlen_zero_should_reject_all_data(data):
    sio = TruncatedStringIO(maxlen=0)
    try:
        sio.write(data)
    except StringTruncated:
        pass
    assert len(sio.getvalue()) == 0, f"maxlen=0 should not store any data, but got {repr(sio.getvalue())}"
```

**Failing input**: Any non-empty string, e.g., `"a"`

## Reproducing the Bug

```python
from Cython.Debugger.libpython import TruncatedStringIO, StringTruncated

sio = TruncatedStringIO(maxlen=0)
sio.write("hello")

print(f"Value: {repr(sio.getvalue())}")
print(f"Length: {len(sio.getvalue())}")
```

Output:
```
Value: 'hello'
Length: 5
```

Expected: Either raise `StringTruncated` immediately, or store empty string.

## Why This Is A Bug

The `TruncatedStringIO` class is designed to limit the amount of data stored to a maximum length. Setting `maxlen=0` is a valid configuration that should prevent any data from being stored. However, due to a falsy check (`if self.maxlen:`), when `maxlen=0`, the value `0` is treated as falsy and the length check is completely skipped, allowing unlimited data to be written.

This violates the documented behavior and the principle of least surprise - a user explicitly setting `maxlen=0` expects no data to be accepted.

## Fix

```diff
--- a/Cython/Debugger/libpython.py
+++ b/Cython/Debugger/libpython.py
@@ -154,7 +154,7 @@ class TruncatedStringIO:
         self.maxlen = maxlen

     def write(self, data):
-        if self.maxlen:
+        if self.maxlen is not None:
             if len(data) + len(self._val) > self.maxlen:
                 # Truncation:
                 self._val += data[0:self.maxlen - len(self._val)]
```

The fix changes the falsy check to an explicit `is not None` check, allowing `maxlen=0` to be properly enforced while still allowing `maxlen=None` to mean "no limit".