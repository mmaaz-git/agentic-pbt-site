# Bug Report: numpy.rec.recarray Field Name Conflicts with Methods

**Target**: `numpy.rec.recarray` (and related functions: `fromrecords`, `fromarrays`, `array`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `recarray` class fails to provide attribute access to fields when the field name conflicts with existing method names (e.g., 'field', 'item', 'copy', 'view', etc.), violating its documented API contract that "allows field access using attributes".

## Property-Based Test

```python
import numpy as np
import numpy.rec
from hypothesis import given, strategies as st
import pytest

RECARRAY_METHOD_NAMES = [
    'field', 'item', 'copy', 'view', 'tolist', 'fill', 'all', 'any',
    'argmax', 'argmin', 'argsort', 'astype', 'byteswap', 'choose', 'clip',
    'compress', 'conj', 'conjugate', 'cumprod', 'cumsum', 'diagonal', 'dot',
    'dump', 'dumps', 'flatten', 'getfield', 'max', 'mean', 'min', 'nonzero',
    'partition', 'prod', 'ptp', 'put', 'ravel', 'repeat', 'reshape', 'resize',
    'round', 'searchsorted', 'setfield', 'setflags', 'sort', 'squeeze', 'std',
    'sum', 'swapaxes', 'take', 'tobytes', 'trace', 'transpose', 'var'
]

@given(st.sampled_from(RECARRAY_METHOD_NAMES),
       st.lists(st.integers(), min_size=1, max_size=10))
def test_method_name_fields_accessible_via_attribute(method_name, data):
    rec = numpy.rec.fromrecords([(x,) for x in data], names=method_name)

    dict_access = rec[method_name]

    attr_access = getattr(rec, method_name)

    if isinstance(attr_access, np.ndarray):
        np.testing.assert_array_equal(attr_access, dict_access)
    else:
        pytest.fail(f"Field '{method_name}': rec.{method_name} returned {type(attr_access).__name__} instead of field data. "
                   f"Dictionary access rec['{method_name}'] works correctly, but attribute access rec.{method_name} returns a method.")
```

**Failing input**: `method_name='field', data=[0]` (or any method name and data)

## Reproducing the Bug

```python
import numpy.rec

rec = numpy.rec.fromrecords([(1,), (2,), (3,)], names='field')

print(rec['field'])

print(rec.field)

try:
    value = rec.field[0]
except TypeError as e:
    print(f"TypeError: {e}")
```

Output:
```
[1 2 3]
<bound method recarray.field of rec.array([(1,), (2,), (3,)], dtype=[('field', '<i8')])>
TypeError: 'method' object is not subscriptable
```

The bug also affects many other common field names: 'item', 'copy', 'view', 'fill', 'all', 'any', 'sum', 'max', 'min', and approximately 50+ method names.

## Why This Is A Bug

The `recarray` documentation explicitly states:

> "Construct an ndarray that allows field access using attributes. [...] Record arrays allow the fields to be accessed as members of the array, using `arr.x` and `arr.y`."

When a field name matches a method name, attribute access (`rec.fieldname`) returns the method instead of field data, violating this documented behavior. Dictionary access (`rec['fieldname']`) works correctly, creating an inconsistency in the API.

This affects real-world usage where users might have natural field names like 'item', 'copy', 'view', 'data', 'sum', 'max', 'min' that happen to conflict with ndarray methods. The API documentation doesn't warn about reserved names.

## Fix

This is a fundamental design issue with Python's attribute access mechanism. The recarray class uses `__getattribute__` to provide field access, but methods take precedence. Possible solutions:

1. **Override `__getattribute__`** to check for field names first before falling back to methods
2. **Document the limitation** and provide a list of reserved field names
3. **Provide a warning** when creating a recarray with conflicting field names

A minimal fix would override `__getattribute__`:

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -XXX,X +XXX,X @@ class recarray(ndarray):
+    def __getattribute__(self, attr):
+        # Check if this is a field name first, before checking for methods
+        try:
+            dtype = object.__getattribute__(self, 'dtype')
+            if dtype.names and attr in dtype.names:
+                return self[attr]
+        except AttributeError:
+            pass
+        # Fall back to normal attribute access for methods
+        return object.__getattribute__(self, attr)
```

However, this may have unintended consequences and requires careful implementation to avoid breaking existing behavior and performance.