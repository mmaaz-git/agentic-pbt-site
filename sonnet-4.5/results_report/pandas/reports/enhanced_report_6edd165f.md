# Bug Report: pandas.plotting register/deregister are not true inverses

**Target**: `pandas.plotting.register_matplotlib_converters` and `pandas.plotting.deregister_matplotlib_converters`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deregister_matplotlib_converters()` function fails to restore matplotlib's unit registry to its original state after `register_matplotlib_converters()` has been called, violating its documented contract. After the first register/deregister cycle, datetime-related converters remain permanently in the registry.

## Property-Based Test

```python
import copy
import matplotlib.units as munits
from hypothesis import given, settings, strategies as st, example
from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
import datetime


def test_register_deregister_inverse():
    """Test that single register/deregister cycle restores original state"""
    original_registry = copy.copy(munits.registry)

    # Show initial state
    had_datetime = datetime.datetime in original_registry
    print(f"  Initial: datetime.datetime in registry = {had_datetime}")

    register_matplotlib_converters()
    after_register = copy.copy(munits.registry)
    print(f"  After register: datetime.datetime in registry = {datetime.datetime in after_register}")

    deregister_matplotlib_converters()
    after_deregister = copy.copy(munits.registry)
    print(f"  After deregister: datetime.datetime in registry = {datetime.datetime in after_deregister}")

    # Check if registry was restored
    if original_registry != after_deregister:
        print(f"  Registry not restored! Original had {len(original_registry)} converters, now has {len(after_deregister)}")
        # Find differences
        for key in set(list(original_registry.keys()) + list(after_deregister.keys())):
            if key not in original_registry:
                print(f"    Added: {key}")
            elif key not in after_deregister:
                print(f"    Removed: {key}")
            elif original_registry[key] != after_deregister[key]:
                print(f"    Changed: {key}")

    assert original_registry == after_deregister, "Registry was not restored to original state"


@given(st.integers(min_value=1, max_value=5))
@example(1)  # Force testing with n=1
@example(2)  # Force testing with n=2
@settings(max_examples=10, deadline=None)
def test_register_deregister_multiple_times(n):
    """Test that multiple register/deregister cycles restore original state"""
    original_registry = copy.copy(munits.registry)

    # Show initial state
    had_datetime = datetime.datetime in original_registry
    print(f"  Testing with n={n}, initial datetime in registry = {had_datetime}")

    for i in range(n):
        register_matplotlib_converters()
        print(f"    After register #{i+1}: datetime.datetime in registry = {datetime.datetime in munits.registry}")

    for i in range(n):
        deregister_matplotlib_converters()
        print(f"    After deregister #{i+1}: datetime.datetime in registry = {datetime.datetime in munits.registry}")

    after_all = copy.copy(munits.registry)

    # Check if registry was restored
    if original_registry != after_all:
        print(f"  Registry not restored! Original had {len(original_registry)} converters, now has {len(after_all)}")

    assert original_registry == after_all, f"Registry was not restored after {n} cycles"


# Run the tests
print("Test 1: Single register/deregister cycle")
print("-" * 40)
try:
    test_register_deregister_inverse()
    print("PASSED")
except AssertionError as e:
    print(f"FAILED: {e}")

print("\nTest 2: Multiple register/deregister cycles with Hypothesis")
print("-" * 40)
try:
    test_register_deregister_multiple_times()
    print("ALL TESTS PASSED")
except AssertionError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")
```

<details>

<summary>
**Failing input**: `n=1` (single register/deregister cycle)
</summary>
```
Test 1: Single register/deregister cycle
----------------------------------------
  Initial: datetime.datetime in registry = False
  After register: datetime.datetime in registry = True
  After deregister: datetime.datetime in registry = True
  Registry not restored! Original had 1 converters, now has 4
    Added: <class 'datetime.date'>
    Added: <class 'numpy.datetime64'>
    Added: <class 'datetime.datetime'>
FAILED: Registry was not restored to original state

Test 2: Multiple register/deregister cycles with Hypothesis
----------------------------------------
  Testing with n=1, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
  Testing with n=2, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After register #2: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
    After deregister #2: datetime.datetime in registry = True
  Testing with n=1, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
  Testing with n=3, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After register #2: datetime.datetime in registry = True
    After register #3: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
    After deregister #2: datetime.datetime in registry = True
    After deregister #3: datetime.datetime in registry = True
  Testing with n=2, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After register #2: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
    After deregister #2: datetime.datetime in registry = True
  Testing with n=5, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After register #2: datetime.datetime in registry = True
    After register #3: datetime.datetime in registry = True
    After register #4: datetime.datetime in registry = True
    After register #5: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
    After deregister #2: datetime.datetime in registry = True
    After deregister #3: datetime.datetime in registry = True
    After deregister #4: datetime.datetime in registry = True
    After deregister #5: datetime.datetime in registry = True
  Testing with n=4, initial datetime in registry = True
    After register #1: datetime.datetime in registry = True
    After register #2: datetime.datetime in registry = True
    After register #3: datetime.datetime in registry = True
    After register #4: datetime.datetime in registry = True
    After deregister #1: datetime.datetime in registry = True
    After deregister #2: datetime.datetime in registry = True
    After deregister #3: datetime.datetime in registry = True
    After deregister #4: datetime.datetime in registry = True
ALL TESTS PASSED
```
</details>

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

<details>

<summary>
Output shows converters not removed after deregister
</summary>
```
Initial state:
datetime.datetime in registry: False

First cycle:
After register: True
After deregister: True

Second cycle:
After register: True
After deregister: True
```
</details>

## Why This Is A Bug

The documentation for `deregister_matplotlib_converters()` explicitly states:

> "Removes the custom converters added by :func:`register`. This attempts to set the state of the registry back to the state before pandas registered its own units. Converters for pandas' own types like Timestamp and Period are removed completely. Converters for types pandas overwrites, like ``datetime.datetime``, are restored to their original value."

This documented behavior is violated in two ways:

1. **Immediate failure**: Even on the first deregister call, the registry is not restored to its original state. The test shows the registry initially has 1 converter, but after register/deregister it has 4 converters, with `datetime.datetime`, `datetime.date`, and `numpy.datetime64` permanently added.

2. **State accumulation**: The bug occurs because the module-level `_mpl_units` dictionary in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/plotting/_matplotlib/converter.py` (line 72) is never cleared between register/deregister cycles.

The deregister function (lines 133-145) attempts to restore converters from `_mpl_units`, but this cache persists across function calls. When converters are already present in matplotlib's registry from a previous cycle, they get cached again, causing them to be permanently restored on subsequent deregister calls.

## Relevant Context

The bug manifests differently depending on matplotlib's initial state:
- When matplotlib starts with no converters (clean state), the bug adds converters that weren't there originally
- When matplotlib already has converters (from previous operations), those converters become "stuck" and cannot be removed

This affects any application that:
- Dynamically manages matplotlib converters
- Uses pandas plotting in plugin architectures
- Needs to ensure clean matplotlib state between operations
- Tests plotting functionality with different converter configurations

Documentation references:
- `deregister_matplotlib_converters` docstring: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/plotting/_misc.py:113-152`
- Implementation: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/plotting/_matplotlib/converter.py:122-145`

## Proposed Fix

Clear the `_mpl_units` cache at the start of each `register()` call to ensure it only contains converters from the current registration cycle:

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -120,6 +120,7 @@ def pandas_converters() -> Generator[None, None, None]:


 def register() -> None:
+    _mpl_units.clear()  # Clear cache to ensure idempotent behavior
     pairs = get_pairs()
     for type_, cls in pairs:
         # Cache previous converter if present
```

This ensures each register/deregister cycle operates independently and `deregister()` correctly restores the state that existed before the most recent `register()` call.