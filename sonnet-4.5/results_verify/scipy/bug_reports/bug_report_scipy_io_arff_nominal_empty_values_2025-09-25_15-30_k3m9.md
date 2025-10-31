# Bug Report: scipy.io.arff NominalAttribute Empty Values Crash

**Target**: `scipy.io.arff._arffread.NominalAttribute.__init__`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`NominalAttribute.__init__` crashes with ValueError when initialized with an empty values list due to calling `max()` on an empty sequence at line 103.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.io.arff._arffread import NominalAttribute
import pytest

@given(name=st.text(min_size=1, max_size=20))
def test_nominal_attribute_empty_values_should_not_crash(name):
    """NominalAttribute should either accept empty values or raise a descriptive error"""
    assume(name.strip() != '')

    try:
        attr = NominalAttribute(name, ())
        assert hasattr(attr, 'dtype')
    except ValueError as e:
        assert 'empty sequence' in str(e).lower()
```

**Failing input**: Any attribute name with empty values tuple, e.g., `NominalAttribute("test", ())`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import NominalAttribute

try:
    attr = NominalAttribute("test_attr", ())
except ValueError as e:
    print(f"Crash: {e}")
```

Output:
```
Crash: max() arg is an empty sequence
```

This can also be triggered by ARFF files with empty nominal attribute definitions:
```python
from io import StringIO
from scipy.io.arff import loadarff

arff_content = """@RELATION test
@ATTRIBUTE color {}
@DATA
"""

try:
    data, meta = loadarff(StringIO(arff_content))
except ValueError as e:
    print(f"Crash when loading ARFF: {e}")
```

## Why This Is A Bug

Line 103 in `_arffread.py`:
```python
self.dtype = (np.bytes_, max(len(i) for i in values))
```

When `values` is an empty tuple or list, `max(len(i) for i in values)` attempts to find the maximum of an empty sequence, raising `ValueError: max() arg is an empty sequence`.

This violates the robustness principle - the class should either:
1. Accept empty nominal values (with appropriate dtype), or
2. Raise a descriptive `ParseArffError` explaining that nominal attributes must have at least one value

Currently it crashes with a confusing low-level error that doesn't explain the actual problem.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -99,7 +99,10 @@ class NominalAttribute(Attribute):
     def __init__(self, name, values):
         super().__init__(name)
         self.values = values
         self.range = values
-        self.dtype = (np.bytes_, max(len(i) for i in values))
+        if not values:
+            raise ParseArffError(f"Nominal attribute '{name}' must have at least one possible value")
+        self.dtype = (np.bytes_, max(len(i) for i in values))
```

This fix provides a clear, descriptive error message when empty nominal values are encountered, making it easier for users to understand and fix the problem in their ARFF files.