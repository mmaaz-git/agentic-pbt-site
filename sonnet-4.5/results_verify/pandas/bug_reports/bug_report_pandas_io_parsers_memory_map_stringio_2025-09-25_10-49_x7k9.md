# Bug Report: pandas.io.parsers memory_map with StringIO

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Using `memory_map=True` with `io.StringIO` causes an unhelpful `UnsupportedOperation` error instead of either working gracefully or providing a clear error message about the limitation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import io
import pandas as pd

@given(
    data=st.lists(
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=5),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=50)
def test_memory_map_does_not_affect_output(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    csv_content = '\n'.join(','.join(map(str, row)) for row in data)

    df_with_mmap = pd.read_csv(io.StringIO(csv_content), header=None, memory_map=True)
    df_without_mmap = pd.read_csv(io.StringIO(csv_content), header=None, memory_map=False)

    pd.testing.assert_frame_equal(df_with_mmap, df_without_mmap)
```

**Failing input**: `data=[[0]]`

## Reproducing the Bug

```python
import io
import pandas as pd

csv_content = "a,b\n1,2\n3,4"
df = pd.read_csv(io.StringIO(csv_content), memory_map=True)
```

**Output:**
```
io.UnsupportedOperation: fileno
```

The error occurs because:
1. `io.StringIO` has a `fileno()` method, so `hasattr(io.StringIO(''), 'fileno')` returns `True`
2. However, calling `fileno()` on a StringIO raises `UnsupportedOperation`
3. pandas `_maybe_memory_map` uses `hasattr` to check if memory mapping is possible
4. This causes the check to pass, but then calling `fileno()` fails with a confusing error

## Why This Is A Bug

The error message "UnsupportedOperation: fileno" is confusing and doesn't explain the real issue: `memory_map` only works with actual files, not in-memory buffers like StringIO.

Expected behavior should be one of:
1. Silently fall back to non-memory-mapped reading when used with incompatible buffers
2. Raise a clear `ValueError` explaining that `memory_map=True` requires a file path or file object with a valid file descriptor
3. Make memory mapping work transparently with all buffer types (likely not feasible)

The current implementation tries to check compatibility with `hasattr(handle, "fileno")`, but this is insufficient because some file-like objects (like StringIO) have the method but it raises an exception when called.

## Fix

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -1128,7 +1128,14 @@ def _maybe_memory_map(
 ) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
     """Try to memory map file/buffer."""
     handles: list[BaseBuffer] = []
-    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
+
+    if not isinstance(handle, str):
+        try:
+            handle.fileno()
+        except (AttributeError, io.UnsupportedOperation):
+            memory_map = False
+
+    memory_map &= True
     if not memory_map:
         return handle, memory_map, handles
```

Alternatively, a simpler fix that just improves the error message:

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -1128,7 +1128,7 @@ def _maybe_memory_map(
 ) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
     """Try to memory map file/buffer."""
     handles: list[BaseBuffer] = []
-    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
+    memory_map &= isinstance(handle, str) or (hasattr(handle, "fileno") and callable(getattr(handle, "fileno", None)))
     if not memory_map:
         return handle, memory_map, handles

@@ -1140,7 +1140,12 @@ def _maybe_memory_map(
         handles.append(handle)

     try:
-        # open mmap and adds *-able
+        try:
+            fileno = handle.fileno()
+        except (AttributeError, io.UnsupportedOperation) as e:
+            raise ValueError(
+                f"memory_map=True requires a file path or file object with a valid file descriptor, "
+                f"not {type(handle).__name__}"
+            ) from e
         # error: Argument 1 to "_IOWrapper" has incompatible type "mmap";
         # expected "BaseBuffer"
         wrapped = _IOWrapper(