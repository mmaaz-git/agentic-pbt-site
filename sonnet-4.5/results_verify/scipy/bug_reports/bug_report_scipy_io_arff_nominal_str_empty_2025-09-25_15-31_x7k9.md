# Bug Report: scipy.io.arff NominalAttribute.__str__ IndexError on Empty Values

**Target**: `scipy.io.arff._arffread.NominalAttribute.__str__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`NominalAttribute.__str__` raises IndexError when the attribute has an empty values list, due to accessing `self.values[-1]` at line 166 without checking if the list is non-empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import NominalAttribute
import numpy as np
import pytest

@given(name=st.text(min_size=1, max_size=20))
def test_nominal_str_should_handle_empty_values(name):
    """__str__ should handle empty values list gracefully"""
    attr = NominalAttribute.__new__(NominalAttribute)
    attr.name = name
    attr.values = ()
    attr.dtype = np.bytes_
    attr.range = ()
    attr.type_name = 'nominal'

    try:
        result = str(attr)
        assert isinstance(result, str)
    except IndexError:
        pytest.fail("__str__ should not raise IndexError on empty values")
```

**Failing input**: Any `NominalAttribute` instance with empty `values` tuple

## Reproducing the Bug

```python
from scipy.io.arff._arffread import NominalAttribute
import numpy as np

attr = NominalAttribute.__new__(NominalAttribute)
attr.name = "test"
attr.values = ()
attr.dtype = np.bytes_
attr.type_name = 'nominal'

try:
    result = str(attr)
    print(f"String representation: {result}")
except IndexError as e:
    print(f"Crash: IndexError - {e}")
```

Output:
```
Crash: IndexError - tuple index out of range
```

## Why This Is A Bug

Lines 161-167 in `_arffread.py`:
```python
def __str__(self):
    msg = self.name + ",{"
    for i in range(len(self.values)-1):
        msg += self.values[i] + ","
    msg += self.values[-1]  # Line 166 - crashes if values is empty
    msg += "}"
    return msg
```

When `self.values` is an empty tuple, `self.values[-1]` raises `IndexError: tuple index out of range`.

While this is a secondary failure (the first bug prevents empty values from being created normally), it still represents a robustness issue where the `__str__` method doesn't handle all possible object states.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -161,9 +161,12 @@ class NominalAttribute(Attribute):
     def __str__(self):
         msg = self.name + ",{"
-        for i in range(len(self.values)-1):
-            msg += self.values[i] + ","
-        msg += self.values[-1]
+        if self.values:
+            for i in range(len(self.values)-1):
+                msg += self.values[i] + ","
+            msg += self.values[-1]
+        # For empty values, just leave the braces empty: "attr,{}"
         msg += "}"
         return msg
```

This fix makes `__str__` more robust by handling the empty values case gracefully, returning a string like `"attr,{}"` instead of crashing.