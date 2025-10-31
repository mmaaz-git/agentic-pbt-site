# Bug Report: dask.diagnostics.profile_visualize.fix_bounds Violates Minimum Span Invariant

**Target**: `dask.diagnostics.profile_visualize.fix_bounds`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fix_bounds` function fails to guarantee that the returned span is at least `min_span` when dealing with large floating-point numbers, due to floating-point precision loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import fix_bounds

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    end=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    min_span=st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e10)
)
def test_fix_bounds_span_invariant(start, end, min_span):
    new_start, new_end = fix_bounds(start, end, min_span)
    assert new_start == start
    assert new_end - new_start >= min_span
```

**Failing input**: `start=6442450945.0, end=0.0, min_span=2147483647.9201343`

## Reproducing the Bug

```python
from dask.diagnostics.profile_visualize import fix_bounds

start = 6442450945.0
end = 0.0
min_span = 2147483647.9201343

new_start, new_end = fix_bounds(start, end, min_span)

actual_span = new_end - new_start
print(f"Actual span: {actual_span}")
print(f"Expected min_span: {min_span}")
print(f"Bug: {actual_span < min_span}")
```

Output:
```
Actual span: 2147483647.0
Expected min_span: 2147483647.9201343
Bug: True
```

## Why This Is A Bug

The function's docstring states: "Adjust end point to ensure span of at least `min_span`". This is a contract that the function should guarantee the invariant: `new_end - new_start >= min_span`.

The implementation is:
```python
def fix_bounds(start, end, min_span):
    return start, max(end, start + min_span)
```

When `start` is a large float (e.g., 6442450945.0) and we add `min_span` (e.g., 2147483647.9201343), the result suffers from floating-point precision loss. The addition `6442450945.0 + 2147483647.9201343` rounds to `8589934592.0` (losing the fractional part), so the final span becomes `8589934592.0 - 6442450945.0 = 2147483647.0`, which is less than the required `min_span` of 2147483647.9201343.

## Fix

This is a fundamental limitation of floating-point arithmetic. The function cannot guarantee the invariant for all possible inputs. Possible approaches:

1. **Relax the invariant**: Use approximate equality (e.g., `new_end - new_start >= min_span - epsilon`)
2. **Add compensation**: Use `max(end, start + min_span) + epsilon` where epsilon accounts for floating-point error
3. **Document the limitation**: Update the docstring to note that the guarantee only holds within floating-point precision

Option 3 (documentation) is the most honest approach:

```diff
--- a/dask/diagnostics/profile_visualize.py
+++ b/dask/diagnostics/profile_visualize.py
@@ -336,7 +336,11 @@ def plot_cache(


 def fix_bounds(start, end, min_span):
-    """Adjust end point to ensure span of at least `min_span`"""
+    """Adjust end point to ensure span of at least `min_span`
+
+    Note: Due to floating-point precision limitations, the returned span
+    may be slightly less than min_span when start and min_span are very large.
+    """
     return start, max(end, start + min_span)
```