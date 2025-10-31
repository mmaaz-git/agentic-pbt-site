# Bug Report: pandas.plotting._misc._Options.get() Method Ignores Parameter Aliases

**Target**: `pandas.plotting._misc._Options.get()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_Options.get()` method fails to resolve parameter aliases (e.g., "x_compat" → "xaxis.compat"), violating the class's documented aliasing behavior that states it "allows for parameter aliasing".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.plotting._misc import _Options


@given(st.booleans())
def test_options_get_handles_aliases(value):
    opts = _Options()
    opts["xaxis.compat"] = value

    result_canonical = opts.get("xaxis.compat", "default")
    result_alias = opts.get("x_compat", "default")

    assert result_canonical == value
    assert result_alias == value


# Run the test
test_options_get_handles_aliases()
```

<details>

<summary>
**Failing input**: `value=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 18, in <module>
    test_options_get_handles_aliases()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 6, in test_options_get_handles_aliases
    def test_options_get_handles_aliases(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 14, in test_options_get_handles_aliases
    assert result_alias == value
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_options_get_handles_aliases(
    value=False,
)
```
</details>

## Reproducing the Bug

```python
from pandas.plotting._misc import _Options

# Create an _Options instance
opts = _Options()

# Set a value using the canonical key
opts["xaxis.compat"] = True

# Try to get the value using both canonical key and alias
result_canonical = opts.get("xaxis.compat", "default_value")
result_alias = opts.get("x_compat", "default_value")

print(f"opts.get('xaxis.compat', 'default_value') = {result_canonical}")
print(f"opts.get('x_compat', 'default_value') = {result_alias}")

# For comparison, show that __getitem__ works correctly with the alias
print(f"opts['x_compat'] = {opts['x_compat']}")

# Show that __contains__ also works correctly with the alias
print(f"'x_compat' in opts = {'x_compat' in opts}")
```

<details>

<summary>
Output demonstrating the inconsistency between get() and other dict operations
</summary>
```
opts.get('xaxis.compat', 'default_value') = True
opts.get('x_compat', 'default_value') = default_value
opts['x_compat'] = True
'x_compat' in opts = True
```
</details>

## Why This Is A Bug

The `_Options` class documentation (lines 611-614 in pandas/plotting/_misc.py) explicitly states: "Allows for parameter aliasing so you can just use parameter names that are the same as the plot function parameters, but is stored in a canonical format that makes it easy to breakdown into groups later."

The class maintains an `_ALIASES` dictionary mapping `{"x_compat": "xaxis.compat"}` and implements a `_get_canonical_key()` method to translate aliases. All overridden dict methods (`__getitem__`, `__setitem__`, `__contains__`, and `__delitem__`) correctly call `_get_canonical_key()` to resolve aliases before operation.

However, the `get()` method is not overridden and inherits directly from dict, bypassing alias resolution. This creates an inconsistent API where:
- `opts["x_compat"]` correctly returns the value (via `__getitem__`)
- `"x_compat" in opts` correctly returns True (via `__contains__`)
- `opts.get("x_compat")` incorrectly returns the default value (inherited dict.get)

The class's `use()` method documentation also states "Aliasing allowed" (line 678), and the docstring example shows using `"x_compat"` directly with `pd.plotting.plot_params.use()`. This inconsistency violates the principle of least surprise and the documented contract.

## Relevant Context

The `_Options` class is technically internal (underscore prefix) but is exposed publicly via `pd.plotting.plot_params` (line 688). Users interacting with plot parameters would reasonably expect consistent behavior across all dict-like operations.

Verification shows that `get()` is not in `_Options.__dict__` (returns False) and `_Options.get is dict.get` (returns True), confirming the method is inherited without override.

The workaround is to use bracket notation `opts["x_compat"]` instead of `opts.get("x_compat", default)`, but this removes the ability to provide default values for missing keys.

## Proposed Fix

```diff
--- a/pandas/plotting/_misc.py
+++ b/pandas/plotting/_misc.py
@@ -657,6 +657,10 @@ class _Options(dict):
         key = self._get_canonical_key(key)
         return super().__contains__(key)

+    def get(self, key, default=None):
+        key = self._get_canonical_key(key)
+        return super().get(key, default)
+
     def reset(self) -> None:
         """
         Reset the option store to its initial state
```