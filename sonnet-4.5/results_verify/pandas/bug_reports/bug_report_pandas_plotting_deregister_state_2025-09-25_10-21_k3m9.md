# Bug Report: pandas.plotting deregister_matplotlib_converters Fails to Restore Original State

**Target**: `pandas.plotting.deregister_matplotlib_converters`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`deregister_matplotlib_converters()` does not properly restore the matplotlib units registry to its original state before `register_matplotlib_converters()` was called. It leaves 3 extra converters in the registry that were not there initially.

## Property-Based Test

```python
import pytest
import pandas.plotting
import matplotlib.units as munits


def test_deregister_restores_original_state():
    """
    Property: deregister_matplotlib_converters should restore the matplotlib
    registry to its original state before register was called.

    This is explicitly stated in the docstring: "This attempts to set the
    state of the registry back to the state before pandas registered its
    own units."
    """
    initial_keys = set(munits.registry.keys())

    pandas.plotting.register_matplotlib_converters()
    pandas.plotting.deregister_matplotlib_converters()

    after_deregister_keys = set(munits.registry.keys())

    assert after_deregister_keys == initial_keys, (
        f"deregister() should restore the original registry state, but "
        f"initial keys: {sorted([str(k) for k in initial_keys])}, "
        f"after deregister keys: {sorted([str(k) for k in after_deregister_keys])}"
    )
```

**Failing input**: Any call sequence: `register() -> deregister()`

## Reproducing the Bug

```python
import matplotlib.units as munits
import pandas.plotting

initial_keys = set(munits.registry.keys())
print(f"Initial registry has {len(initial_keys)} converter(s):")
for key in sorted([str(k) for k in initial_keys]):
    print(f"  {key}")

pandas.plotting.register_matplotlib_converters()
after_register_keys = set(munits.registry.keys())
print(f"\nAfter register: {len(after_register_keys)} converters")

pandas.plotting.deregister_matplotlib_converters()
after_deregister_keys = set(munits.registry.keys())
print(f"After deregister: {len(after_deregister_keys)} converters")

extra = after_deregister_keys - initial_keys
print(f"\nExtra converters ({len(extra)}):")
for key in sorted([str(k) for k in extra]):
    print(f"  {key}")
```

Output:
```
Initial registry has 1 converter(s):
  <class 'decimal.Decimal'>

After register: 7 converters
After deregister: 4 converters

Extra converters (3):
  <class 'datetime.date'>
  <class 'datetime.datetime'>
  <class 'numpy.datetime64'>
```

## Why This Is A Bug

The docstring for `deregister_matplotlib_converters()` explicitly states: "This attempts to set the state of the registry back to the state before pandas registered its own units."

However, the function adds 3 converters (`datetime.date`, `datetime.datetime`, `numpy.datetime64`) that were not in the registry before pandas registered its converters. This violates the documented behavior and breaks the round-trip property that `register() -> deregister()` should be a no-op.

The root cause is that:
1. When `register()` is called, matplotlib may lazily register default converters for datetime types
2. `register()` caches these matplotlib converters in `_mpl_units`
3. When `deregister()` is called, it restores all converters from `_mpl_units`
4. This restores converters that weren't in the registry before pandas was involved

## Fix

The `deregister()` function should only restore converters that were actually in the registry before `register()` was called. One approach:

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -15,6 +15,7 @@
 from pandas._libs.tslibs.offsets import DateOffset

 _mpl_units = {}
+_initial_registry_keys = None


 def register() -> None:
+    global _initial_registry_keys
+    # Save the initial registry keys before any registration
+    if _initial_registry_keys is None:
+        _initial_registry_keys = set(munits.registry.keys())
+
     pairs = get_pairs()
     for type_, cls in pairs:
         # Cache previous converter if present
@@ -28,11 +33,14 @@
 def deregister() -> None:
     # Renamed in pandas.plotting.__init__
     for type_, cls in get_pairs():
         # We use type to catch our classes directly, no inheritance
         if type(munits.registry.get(type_)) is cls:
             munits.registry.pop(type_)

     # restore the old keys
     for unit, formatter in _mpl_units.items():
+        # Only restore if the unit was in the registry initially
+        if _initial_registry_keys is not None and unit not in _initial_registry_keys:
+            continue
         if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
             # make it idempotent by excluding ours.
             munits.registry[unit] = formatter
```

Alternatively, a simpler fix would be to not restore converters that were not originally in the registry:

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -15,10 +15,13 @@
 from pandas._libs.tslibs.offsets import DateOffset

 _mpl_units = {}
+_was_registered = {}


 def register() -> None:
     pairs = get_pairs()
     for type_, cls in pairs:
+        # Track whether this type was already registered
+        _was_registered[type_] = type_ in munits.registry
         # Cache previous converter if present
         if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
             previous = munits.registry[type_]
@@ -36,6 +39,9 @@

     # restore the old keys
     for unit, formatter in _mpl_units.items():
+        # Don't restore if the unit was not originally registered
+        if not _was_registered.get(unit, False):
+            continue
         if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
             # make it idempotent by excluding ours.
             munits.registry[unit] = formatter
```