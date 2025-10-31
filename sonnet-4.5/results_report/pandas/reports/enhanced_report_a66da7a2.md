# Bug Report: pandas.plotting._misc._Options Methods Bypass Default Key Protection

**Target**: `pandas.plotting._misc._Options.pop()`, `popitem()`, and `clear()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_Options` class enforces a contract that default keys cannot be removed via `__delitem__`, but the inherited `pop()`, `popitem()`, and `clear()` methods bypass this protection, allowing removal of default keys and violating the class invariant.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.plotting._misc import _Options
import pytest


@given(st.booleans())
def test_options_pop_protects_default_keys(value):
    """Test that _Options.pop() raises ValueError when attempting to remove default keys."""
    opts = _Options()
    opts["x_compat"] = value  # Set the value (using the alias)

    # This should raise ValueError because 'xaxis.compat' is a default key
    # But it doesn't - the test will fail showing the bug
    with pytest.raises(ValueError, match="Cannot remove default parameter"):
        opts.pop("xaxis.compat")


if __name__ == "__main__":
    # Run the test with a specific example to demonstrate the failure
    try:
        test_options_pop_protects_default_keys(False)
        print("Test passed (unexpected - there should be a bug)")
    except AssertionError as e:
        print(f"Test failed as expected, demonstrating the bug: {e}")
```

<details>

<summary>
**Failing input**: `False`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/1
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_options_pop_protects_default_keys FAILED                   [100%]

=================================== FAILURES ===================================
____________________ test_options_pop_protects_default_keys ____________________

    @given(st.booleans())
>   def test_options_pop_protects_default_keys(value):
                   ^^^

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

value = False

    @given(st.booleans())
    def test_options_pop_protects_default_keys(value):
        """Test that _Options.pop() raises ValueError when attempting to remove default keys."""
        opts = _Options()
        opts["x_compat"] = value  # Set the value (using the alias)

        # This should raise ValueError because 'xaxis.compat' is a default key
        # But it doesn't - the test will fail showing the bug
>       with pytest.raises(ValueError, match="Cannot remove default parameter"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Falsifying example: test_options_pop_protects_default_keys(
E           value=False,
E       )

hypo.py:14: Failed
=========================== short test summary info ============================
FAILED hypo.py::test_options_pop_protects_default_keys - Failed: DID NOT RAIS...
============================== 1 failed in 0.35s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.plotting._misc import _Options

# Demonstrate the bug where pop() bypasses default key protection
opts = _Options()
print(f"Initial state: {dict(opts)}")
print(f"Default keys: {opts._DEFAULT_KEYS}")

# Try to remove default key using pop() - this should raise ValueError but doesn't
print("\n1. Attempting to remove default key 'xaxis.compat' using pop():")
try:
    removed_value = opts.pop("xaxis.compat")
    print(f"   SUCCESS: pop() removed the key, returned value: {removed_value}")
    print(f"   State after pop: {dict(opts)}")
except ValueError as e:
    print(f"   FAILED: pop() raised ValueError: {e}")

# Reset and try with del for comparison
print("\n2. Resetting and attempting to remove default key using del:")
opts.reset()
print(f"   State after reset: {dict(opts)}")
try:
    del opts["xaxis.compat"]
    print(f"   SUCCESS: del removed the key (unexpected)")
    print(f"   State after del: {dict(opts)}")
except ValueError as e:
    print(f"   FAILED: del correctly raised ValueError: {e}")

# Also test popitem() and clear() which may have the same issue
print("\n3. Testing other dict methods that can remove keys:")
opts.reset()
print(f"   State after reset: {dict(opts)}")

# Test popitem()
print("\n   Testing popitem():")
try:
    key, value = opts.popitem()
    print(f"   SUCCESS: popitem() removed ({key}, {value})")
    print(f"   State after popitem: {dict(opts)}")
except ValueError as e:
    print(f"   FAILED: popitem() raised ValueError: {e}")

# Test clear()
opts.reset()
print("\n   Testing clear():")
try:
    opts.clear()
    print(f"   SUCCESS: clear() removed all keys")
    print(f"   State after clear: {dict(opts)}")
except ValueError as e:
    print(f"   FAILED: clear() raised ValueError: {e}")
```

<details>

<summary>
Inconsistent behavior: pop(), popitem(), and clear() bypass protection while del correctly raises error
</summary>
```
Initial state: {'xaxis.compat': False}
Default keys: ['xaxis.compat']

1. Attempting to remove default key 'xaxis.compat' using pop():
   SUCCESS: pop() removed the key, returned value: False
   State after pop: {}

2. Resetting and attempting to remove default key using del:
   State after reset: {'xaxis.compat': False}
   FAILED: del correctly raised ValueError: Cannot remove default parameter xaxis.compat

3. Testing other dict methods that can remove keys:
   State after reset: {'xaxis.compat': False}

   Testing popitem():
   SUCCESS: popitem() removed (xaxis.compat, False)
   State after popitem: {}

   Testing clear():
   SUCCESS: clear() removed all keys
   State after clear: {}
```
</details>

## Why This Is A Bug

The `_Options` class maintains a clear invariant: default keys listed in `_DEFAULT_KEYS` should not be removable. This is explicitly enforced in the `__delitem__` method (line 650-654 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py`):

```python
def __delitem__(self, key) -> None:
    key = self._get_canonical_key(key)
    if key in self._DEFAULT_KEYS:
        raise ValueError(f"Cannot remove default parameter {key}")
    super().__delitem__(key)
```

However, the class inherits from `dict` without overriding the `pop()`, `popitem()`, and `clear()` methods. These methods call the base dict implementation directly, completely bypassing the protection check. This creates inconsistent behavior:

- `del opts["xaxis.compat"]` → Raises `ValueError` (correct behavior)
- `opts.pop("xaxis.compat")` → Succeeds and removes the key (incorrect)
- `opts.popitem()` → Succeeds and removes the key when it's the only key (incorrect)
- `opts.clear()` → Succeeds and removes all keys including defaults (incorrect)

This violates the principle of least surprise and breaks the class's contract that default parameters cannot be removed.

## Relevant Context

The `_Options` class is used internally by pandas for storing plotting parameters through the `plot_params` instance (line 688). While it's a private class (indicated by the underscore prefix), it's still part of the pandas codebase and should maintain consistent behavior.

The class also implements parameter aliasing (e.g., "x_compat" maps to "xaxis.compat") and includes a context manager `use()` method for temporarily changing options. The protection of default keys ensures that the plotting system always has required parameters available.

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/plotting/_misc.py:608-688`

## Proposed Fix

```diff
--- a/pandas/plotting/_misc.py
+++ b/pandas/plotting/_misc.py
@@ -656,6 +656,22 @@ class _Options(dict):
     def __contains__(self, key) -> bool:
         key = self._get_canonical_key(key)
         return super().__contains__(key)
+
+    def pop(self, key, *args):
+        key = self._get_canonical_key(key)
+        if key in self._DEFAULT_KEYS:
+            raise ValueError(f"Cannot remove default parameter {key}")
+        return super().pop(key, *args)
+
+    def popitem(self):
+        if len(self) == 1 and list(self.keys())[0] in self._DEFAULT_KEYS:
+            raise ValueError(f"Cannot remove default parameter {list(self.keys())[0]}")
+        return super().popitem()
+
+    def clear(self) -> None:
+        if self._DEFAULT_KEYS:
+            raise ValueError("Cannot clear options containing default parameters")
+        super().clear()

     def reset(self) -> None:
         """
```