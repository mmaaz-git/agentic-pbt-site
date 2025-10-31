# Bug Report: pandas.core.util.hashing UnicodeEncodeError with Surrogate Characters

**Target**: `pandas.core.util.hashing._hash_ndarray`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`hash_array()` raises `UnicodeEncodeError` when hashing object arrays containing Unicode surrogate characters (U+D800 to U+DFFF), instead of handling them gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.util.hashing import hash_array


@given(st.lists(st.text(max_size=100, alphabet=st.characters(min_codepoint=128, max_codepoint=0x10FFFF)), min_size=1, max_size=50))
def test_hash_array_unicode_strings(values):
    arr = np.array(values, dtype=object)
    hash1 = hash_array(arr)
    hash2 = hash_array(arr)
    assert np.array_equal(hash1, hash2)
```

**Failing input**: `values=['\ud800']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.util.hashing import hash_array

arr = np.array(['\ud800'], dtype=object)
hash_array(arr)
```

Output:
```
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
```

## Why This Is A Bug

The function crashes on valid Python strings containing surrogate characters. While surrogates are technically invalid in UTF-8, Python allows them in strings (they're used internally for UTF-16 encoding). The function should handle this gracefully rather than raising an unhandled exception.

The code already has a try/except block for `TypeError` at lines 325-331 in hashing.py, but it doesn't catch `UnicodeEncodeError`. This inconsistency suggests error handling was considered but incomplete.

## Fix

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -324,7 +324,7 @@ def _hash_ndarray(

         try:
             vals = hash_object_array(vals, hash_key, encoding)
-        except TypeError:
+        except (TypeError, UnicodeEncodeError):
             # we have mixed types
             vals = hash_object_array(
                 vals.astype(str).astype(object), hash_key, encoding
```

Note: This fix will still raise `UnicodeEncodeError` for surrogates after the str conversion. A more complete fix would use `encoding='utf-8', errors='surrogatepass'` or `errors='replace'` in the hash_object_array call, but that requires modifying the Cython code.

Alternative minimal fix to avoid the crash:

```diff
--- a/pandas/core/util/hashing.py
+++ b/pandas/core/util/hashing.py
@@ -324,10 +324,13 @@ def _hash_ndarray(

         try:
             vals = hash_object_array(vals, hash_key, encoding)
-        except TypeError:
+        except (TypeError, UnicodeEncodeError):
             # we have mixed types
-            vals = hash_object_array(
-                vals.astype(str).astype(object), hash_key, encoding
+            # Use surrogateescape to handle invalid surrogates
+            import sys
+            vals = hash_object_array(
+                np.array([s.encode('utf-8', errors='replace').decode('utf-8')
+                         for s in vals.astype(str)], dtype=object), hash_key, encoding
             )

     # Then, redistribute these 64-bit ints within the space of 64-bit ints
```