# Bug Report: Cython.Build.Inline.cymeit Infinite Loop with Float Timers

**Target**: `Cython.Build.Inline.cymeit` (autorange function at lines 368-380)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `autorange()` inner function in `cymeit` can enter an infinite loop when given a float timer that consistently returns very small values. The sanity check at lines 377-379 only applies to nanosecond (integer) timers, leaving float timers unprotected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=100)
@given(
    timer_value=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
)
def test_cymeit_float_timer_terminates(timer_value):
    from Cython.Build.Inline import cymeit

    counter = [0]
    max_calls = 10000

    def constant_float_timer():
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError(f"Infinite loop detected: timer called {max_calls}+ times")
        return timer_value

    code = "x = 1"

    try:
        cymeit(code, timer=constant_float_timer, repeat=3)
        assert counter[0] < max_calls
    except RuntimeError as e:
        if "Infinite loop detected" in str(e):
            raise AssertionError(f"cymeit entered infinite loop with timer value {timer_value}")
        raise
```

**Failing input**: Any float timer that consistently returns values < 0.2 (min_runtime)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Inline import cymeit

counter = [0]

def zero_float_timer():
    counter[0] += 1
    if counter[0] > 10000:
        raise RuntimeError("INFINITE LOOP: autorange() called timer 10000+ times")
    return 0.0

code = "x = 1"

try:
    cymeit(code, timer=zero_float_timer, repeat=3)
except RuntimeError as e:
    if "INFINITE LOOP" in str(e):
        print(f"BUG: {e}")
        print(f"autorange() has no termination check for float timers")
```

## Why This Is A Bug

At `Cython/Build/Inline.py:377-379`, there is a sanity check to prevent infinite loops:

```python
elif timer_returns_nanoseconds and (time_taken < 10 and number >= 10):
    # Arbitrary sanity check to prevent endless loops for non-ns timers.
    raise RuntimeError(f"Timer seems to return non-ns timings: {timer}")
```

However, this check only applies when `timer_returns_nanoseconds` is `True`. Float timers (where `isinstance(timer(), int)` is `False`) have no such protection.

If a float timer consistently returns values less than `min_runtime` (0.2 seconds), the autorange loop will never terminate:
- Line 366: `min_runtime = one_second / 5 = 1.0 / 5 = 0.2`
- Line 375: `if time_taken >= min_runtime:` - never True if timer always returns < 0.2
- Line 380: `i *= 10` - keeps increasing indefinitely
- No maximum iteration check or timeout for float timers

While real-world timers rarely return consistent zeros, broken timers, mocking in tests, or timer implementation bugs could trigger this infinite loop.

## Fix

```diff
--- a/Cython/Build/Inline.py
+++ b/Cython/Build/Inline.py
@@ -366,14 +366,20 @@ def cymeit(code, setup_code=None, import_module=None, directives=None, timer=ti
     min_runtime = one_second // 5 if timer_returns_nanoseconds else one_second / 5

     def autorange():
         i = 1
+        iterations = 0
         while True:
+            iterations += 1
+            if iterations > 100:
+                raise RuntimeError(
+                    f"Timer appears broken: still returning < {min_runtime} after {iterations} attempts")
             for j in 1, 2, 5:
                 number = i * j
                 time_taken = timeit(number)
                 assert isinstance(time_taken, int if timer_returns_nanoseconds else float)
                 if time_taken >= min_runtime:
                     return number
                 elif timer_returns_nanoseconds and (time_taken < 10 and number >= 10):
```