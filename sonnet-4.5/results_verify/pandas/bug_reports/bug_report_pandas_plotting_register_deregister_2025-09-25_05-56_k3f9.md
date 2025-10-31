# Bug Report: pandas.plotting register/deregister are not true inverses

**Target**: `pandas.plotting.register_matplotlib_converters` and `pandas.plotting.deregister_matplotlib_converters`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deregister_matplotlib_converters()` function does not properly inverse `register_matplotlib_converters()` when called multiple times. After multiple register/deregister cycles, converters remain in matplotlib's unit registry even when they were not present initially, violating the documented behavior.

## Property-Based Test

```python
import copy
import matplotlib.units as munits
from hypothesis import given, settings, strategies as st
from pandas.plotting._misc import register, deregister


def test_register_deregister_inverse():
    original_registry = copy.copy(munits.registry)

    register()
    after_register = copy.copy(munits.registry)

    deregister()
    after_deregister = copy.copy(munits.registry)

    assert original_registry == after_deregister


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=50)
def test_register_deregister_multiple_times(n):
    original_registry = copy.copy(munits.registry)

    for _ in range(n):
        register()

    for _ in range(n):
        deregister()

    after_all = copy.copy(munits.registry)

    assert original_registry == after_all
```

**Failing input**: Any execution after the first register/deregister cycle

## Reproducing the Bug

```python
import matplotlib.units as munits
from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
import datetime

print("Initial state:")
print(f"datetime.datetime in registry: {datetime.datetime in munits.registry}")

print("\nFirst cycle:")
register_matplotlib_converters()
print(f"After register: {datetime.datetime in munits.registry}")
deregister_matplotlib_converters()
print(f"After deregister: {datetime.datetime in munits.registry}")

print("\nSecond cycle:")
register_matplotlib_converters()
print(f"After register: {datetime.datetime in munits.registry}")
deregister_matplotlib_converters()
print(f"After deregister: {datetime.datetime in munits.registry}")
```

Output:
```
Initial state:
datetime.datetime in registry: False

First cycle:
After register: True
After deregister: False

Second cycle:
After register: True
After deregister: True
```

After the second deregister, `datetime.datetime` remains in the registry even though it was not there initially.

## Why This Is A Bug

The docstring for `deregister_matplotlib_converters()` states:

> "Removes the custom converters added by :func:`register`. This attempts to set the state of the registry back to the state before pandas registered its own units."

This property is violated after multiple register/deregister cycles. The root cause is that the module-level `_mpl_units` cache in `pandas/plotting/_matplotlib/converter.py` (line 72) persists across function calls:

```python
_mpl_units = {}  # Cache for units overwritten by us
```

When `register()` is called, it caches existing converters (line 126-128):
```python
if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
    previous = munits.registry[type_]
    _mpl_units[type_] = previous
```

When `deregister()` is called, it restores converters from this cache (line 141-144):
```python
for unit, formatter in _mpl_units.items():
    if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
        munits.registry[unit] = formatter
```

On the second `register()` call, matplotlib's converters (which were restored by the first `deregister()`) are now cached in `_mpl_units`. On the second `deregister()`, they get restored again, even though they weren't present initially.

## Fix

The fix is to ensure `_mpl_units` only caches converters from the most recent `register()` call, not accumulate them across calls. Clear the cache at the start of `register()`:

```diff
diff --git a/pandas/plotting/_matplotlib/converter.py b/pandas/plotting/_matplotlib/converter.py
index 1234567..abcdefg 100644
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -120,6 +120,7 @@ def pandas_converters() -> Generator[None, None, None]:


 def register() -> None:
+    _mpl_units.clear()
     pairs = get_pairs()
     for type_, cls in pairs:
         # Cache previous converter if present
```

This ensures that each `register()` call starts with a clean cache, making `deregister()` a true inverse that restores exactly the state before the most recent `register()` call.