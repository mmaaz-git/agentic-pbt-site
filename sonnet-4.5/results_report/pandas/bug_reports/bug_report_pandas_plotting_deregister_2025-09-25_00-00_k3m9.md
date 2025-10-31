# Bug Report: pandas.plotting.deregister_matplotlib_converters Fails to Restore Registry State

**Target**: `pandas.plotting.deregister_matplotlib_converters`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`deregister_matplotlib_converters()` fails to restore `matplotlib.units.registry` to its pre-registration state, leaving extra converters for `datetime.datetime`, `datetime.date`, and `numpy.datetime64` that were not present before `register_matplotlib_converters()` was called. This violates the documented contract that deregister "attempts to set the state of the registry back to the state before pandas registered its own units."

## Property-Based Test

```python
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.units as munits
from hypothesis import given, strategies as st
import copy

@given(st.integers(min_value=1, max_value=100))
def test_register_deregister_inverse_operation(seed):
    """
    Property: deregister should restore registry state to before register was called.

    Evidence: Docstring states "attempts to set the state of the registry back
    to the state before pandas registered its own units."
    """

    original_registry_keys = set(munits.registry.keys())

    pandas.plotting.register_matplotlib_converters()
    pandas.plotting.deregister_matplotlib_converters()

    restored_registry_keys = set(munits.registry.keys())

    assert restored_registry_keys == original_registry_keys, \
        f"deregister should restore original state. Original: {[k.__name__ for k in original_registry_keys]}, Restored: {[k.__name__ for k in restored_registry_keys]}"
```

**Failing input**: Any input triggers the failure

## Reproducing the Bug

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.units as munits
import pandas.plotting

print("Initial registry:", [k.__name__ for k in munits.registry.keys()])

pandas.plotting.register_matplotlib_converters()
print("After register:", [k.__name__ for k in munits.registry.keys()])

pandas.plotting.deregister_matplotlib_converters()
print("After deregister:", [k.__name__ for k in munits.registry.keys()])
```

**Expected output:**
```
Initial registry: ['Decimal']
After register: ['Decimal', 'Timestamp', 'Period', 'datetime', 'date', 'time', 'datetime64']
After deregister: ['Decimal']
```

**Actual output:**
```
Initial registry: ['Decimal']
After register: ['Decimal', 'Period', 'Timestamp', 'time', 'date', 'datetime64', 'datetime']
After deregister: ['Decimal', 'date', 'datetime', 'datetime64']
```

## Why This Is A Bug

The docstring for `deregister_matplotlib_converters()` explicitly states:

> "Removes the custom converters added by :func:`register`. This attempts to set the state of the registry back to the state before pandas registered its own units."

However, the function leaves `datetime.datetime`, `datetime.date`, and `numpy.datetime64` converters in the registry even when these types were not present before `register()` was called. This violates the documented contract and creates unexpected state mutations.

The issue occurs because:
1. When `register()` is called, if datetime/date/datetime64 are not in the registry, nothing is cached for them
2. During registration, matplotlib's `_SwitchableDateConverter` gets registered for these types (as a side effect)
3. When `deregister()` is called, it caches and then restores these `_SwitchableDateConverter` instances
4. But these converters were never in the original pre-registration state

## Fix

The `deregister()` function in `pandas/plotting/_matplotlib/converter.py` should only restore converters that were present *before* the corresponding `register()` call. Currently it restores all cached converters, including those that were added as side effects during registration.

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -15,6 +15,7 @@
 from matplotlib import units as munits

 _mpl_units = {}
+_original_keys = None

 def register() -> None:
+    global _original_keys
+    _original_keys = set(munits.registry.keys())
     pairs = get_pairs()
     for type_, cls in pairs:
         if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
@@ -30,6 +33,7 @@
         munits.registry[type_] = cls()

 def deregister() -> None:
+    global _original_keys
     for type_, cls in get_pairs():
         if type(munits.registry.get(type_)) is cls:
             munits.registry.pop(type_)

     for unit, formatter in _mpl_units.items():
-        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
+        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter} and \
+           (_original_keys is None or unit in _original_keys):
             munits.registry[unit] = formatter
+    _original_keys = None
```

This ensures that only converters that existed before registration are restored, properly honoring the documented contract.