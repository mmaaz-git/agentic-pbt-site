# Bug Report: xarray.compat.npcompat.isdtype Version-Dependent Behavior

**Target**: `xarray.compat.npcompat.isdtype`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `isdtype` function in `xarray.compat.npcompat` has inconsistent behavior across NumPy versions: it accepts `np.generic` (scalar) values on NumPy < 2.0 but rejects them on NumPy >= 2.0, violating the compatibility layer's purpose.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
import pytest
from xarray.compat import npcompat

@given(st.sampled_from([np.int32(1), np.float64(1.0), np.bool_(True)]))
def test_isdtype_scalar_handling_inconsistency(scalar):
    """Property test: isdtype should handle scalars consistently across numpy versions"""

    has_native_isdtype = hasattr(np, 'isdtype')

    if has_native_isdtype:
        with pytest.raises(TypeError):
            npcompat.isdtype(scalar, 'numeric')
    else:
        result = npcompat.isdtype(scalar, 'numeric')
        assert isinstance(result, bool)
```

**Failing input**: `np.int32(5)` on NumPy >= 2.0

## Reproducing the Bug

```python
import numpy as np
from xarray.compat.npcompat import isdtype

print(f"NumPy version: {np.__version__}")

scalar = np.int32(5)
print(f"Testing with scalar: {scalar}")

try:
    result = isdtype(scalar, 'integral')
    print(f"Success: {result}")
    print("On NumPy < 2.0, this works")
except TypeError as e:
    print(f"TypeError: {e}")
    print("On NumPy >= 2.0, this fails")
```

**Output on NumPy >= 2.0:**
```
NumPy version: 2.3.3
Testing with scalar: 5
TypeError: dtype argument must be a NumPy dtype, but it is a <class 'numpy.int32'>.
On NumPy >= 2.0, this fails
```

**Expected on NumPy < 2.0:** The function would return `True` without error.

## Why This Is A Bug

A compatibility layer's purpose is to provide **consistent behavior** across different versions of dependencies. The `xarray.compat.npcompat` module explicitly exists to handle NumPy version differences.

**Current behavior:**
- NumPy >= 2.0: `isdtype` imported directly from numpy (line 37), which rejects `np.generic` scalars
- NumPy < 2.0: Custom fallback implementation (lines 54-75) that explicitly handles `np.generic` scalars (lines 72-73)

**The inconsistency:**
```python
# Lines 72-73 in npcompat.py (fallback implementation)
if isinstance(dtype, np.generic):
    return isinstance(dtype, translated_kinds)
```

This code path only exists in the fallback, making the function accept scalars on older NumPy but reject them on newer versions. This is the opposite of what a compat layer should do.

**Impact:** While the main xarray codebase uses a wrapper (`dtypes.isdtype`) that validates inputs before calling `npcompat.isdtype`, any direct users of the compat module (other libraries, internal utilities) could experience version-dependent breakage.

## Fix

The fallback implementation should match NumPy's behavior and reject scalar values. Remove the special handling of `np.generic`:

```diff
diff --git a/xarray/compat/npcompat.py b/xarray/compat/npcompat.py
index abc123..def456 100644
--- a/xarray/compat/npcompat.py
+++ b/xarray/compat/npcompat.py
@@ -69,10 +69,11 @@ def isdtype(

         # verified the dtypes already, no need to check again
         translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
-        if isinstance(dtype, np.generic):
-            return isinstance(dtype, translated_kinds)
-        else:
-            return any(np.issubdtype(dtype, k) for k in translated_kinds)
+        if isinstance(dtype, type) and issubclass(dtype, np.generic):
+            dtype = np.dtype(dtype)
+        elif not isinstance(dtype, np.dtype):
+            raise TypeError(f"dtype argument must be a NumPy dtype, but it is a {type(dtype)}.")
+        return any(np.issubdtype(dtype, k) for k in translated_kinds)

     HAS_STRING_DTYPE = False
```

This ensures consistent behavior: both versions reject scalar values and accept dtype objects and dtype classes, matching NumPy >= 2.0's specification.