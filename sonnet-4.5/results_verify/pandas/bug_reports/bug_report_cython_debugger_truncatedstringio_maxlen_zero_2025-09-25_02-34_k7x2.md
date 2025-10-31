# Bug Report: Cython.Debugger.libpython.TruncatedStringIO Maxlen Zero Handling

**Target**: `Cython.Debugger.libpython.TruncatedStringIO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `TruncatedStringIO.write()` method incorrectly treats `maxlen=0` as unlimited writes due to using a truthiness check instead of an explicit None check, violating the class's truncation contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Debugger.libpython import TruncatedStringIO, StringTruncated
import pytest

@given(st.text(min_size=1))
def test_zero_maxlen_immediate_truncation(data):
    tio = TruncatedStringIO(maxlen=0)
    with pytest.raises(StringTruncated):
        tio.write(data)
    result = tio.getvalue()
    assert len(result) == 0
    assert result == ''
```

**Failing input**: Any non-empty string, e.g., `"x"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.libpython import TruncatedStringIO, StringTruncated

tio = TruncatedStringIO(maxlen=0)
tio.write("x")
print(f"getvalue(): '{tio.getvalue()}'")
print(f"Length: {len(tio.getvalue())}")
```

Expected output:
```
StringTruncated exception raised
getvalue(): ''
Length: 0
```

Actual output:
```
getvalue(): 'x'
Length: 1
```

## Why This Is A Bug

The `TruncatedStringIO` class is designed to limit output length and raise `StringTruncated` when the limit is exceeded. However, the condition `if self.maxlen:` on line 157 treats `maxlen=0` as falsy, bypassing all truncation logic. This means:

1. **Violates documented behavior**: The class should truncate at maxlen bytes, but maxlen=0 allows unlimited writes
2. **Breaks invariant**: The fundamental property `len(getvalue()) <= maxlen` is violated
3. **Inconsistent semantics**: `maxlen=0` should mean "allow zero bytes" but instead means "allow unlimited bytes"

This makes `maxlen=0` and `maxlen=None` behave identically, which is semantically incorrect.

## Fix

```diff
--- a/Cython/Debugger/libpython.py
+++ b/Cython/Debugger/libpython.py
@@ -155,7 +155,7 @@ class TruncatedStringIO:
         self.maxlen = maxlen

     def write(self, data):
-        if self.maxlen:
+        if self.maxlen is not None:
             if len(data) + len(self._val) > self.maxlen:
                 # Truncation:
                 self._val += data[0:self.maxlen - len(self._val)]
```