# Bug Report: pandas.plotting._misc._Options pop() Method Bypasses Default Key Protection

**Target**: `pandas.plotting._misc._Options.pop()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_Options.pop()` method allows removal of default keys, violating the class's invariant that default keys cannot be deleted. While `__delitem__` correctly raises `ValueError` when attempting to remove default keys, `pop()` bypasses this protection by calling the inherited dict.pop() directly.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.plotting._misc import _Options
import pytest


@given(st.booleans())
def test_options_pop_protects_default_keys(value):
    opts = _Options()
    opts["x_compat"] = value

    with pytest.raises(ValueError, match="Cannot remove default parameter"):
        opts.pop("xaxis.compat")
```

**Failing input**: Any boolean value (e.g., `False`)

## Reproducing the Bug

```python
from pandas.plotting._misc import _Options

opts = _Options()
print(f"Initial state: {dict(opts)}")

removed_value = opts.pop("xaxis.compat")
print(f"Removed value: {removed_value}")
print(f"State after pop: {dict(opts)}")

print("\nComparison - del correctly raises error:")
opts2 = _Options()
try:
    del opts2["xaxis.compat"]
    print("del succeeded (unexpected)")
except ValueError as e:
    print(f"del raised ValueError: {e}")
```

**Output:**
```
Initial state: {'xaxis.compat': False}
Removed value: False
State after pop: {}

Comparison - del correctly raises error:
del raised ValueError: Cannot remove default parameter xaxis.compat
```

## Why This Is A Bug

The `_Options` class maintains an invariant that default keys (listed in `_DEFAULT_KEYS`) cannot be removed. This is enforced in `__delitem__`:

```python
def __delitem__(self, key) -> None:
    key = self._get_canonical_key(key)
    if key in self._DEFAULT_KEYS:
        raise ValueError(f"Cannot remove default parameter {key}")
    super().__delitem__(key)
```

However, `pop()` is inherited from dict without overriding, so it bypasses this check and calls `dict.pop()` directly. This creates inconsistent behavior:
- `del opts["xaxis.compat"]` raises ValueError (correct)
- `opts.pop("xaxis.compat")` succeeds and removes the key (incorrect)

This breaks the class invariant and can leave the options object in an invalid state.

## Fix

Override the `pop()` method to apply the same protection as `__delitem__`:

```diff
--- a/pandas/plotting/_misc.py
+++ b/pandas/plotting/_misc.py
@@ -60,6 +60,14 @@ class _Options(dict):
         key = self._get_canonical_key(key)
         return super().__contains__(key)

+    def pop(self, key, *args):
+        key = self._get_canonical_key(key)
+        if key in self._DEFAULT_KEYS:
+            raise ValueError(f"Cannot remove default parameter {key}")
+        return super().pop(key, *args)
+
     def reset(self) -> None:
         """
         Reset the option store to its initial state
```