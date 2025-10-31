# Bug Report: pandas.core.strings.accessor.StringMethods.slice_replace - Incorrect behavior when start >= stop

**Target**: `pandas.core.strings.accessor.StringMethods.slice_replace`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `slice_replace()` method incorrectly handles cases where `start >= stop`, deviating from both its documented behavior and Python's standard slicing semantics by ignoring the `stop` parameter entirely.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(min_size=1), min_size=1, max_size=100), st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10), st.text())
@settings(max_examples=1000)
def test_slice_replace_consistency(strings, start, stop, repl):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if pd.notna(s.iloc[i]) and pd.notna(replaced.iloc[i]):
            expected = s.iloc[i][:start] + repl + s.iloc[i][stop:]
            assert replaced.iloc[i] == expected, f"Failed for string='{s.iloc[i]}', start={start}, stop={stop}, repl='{repl}'. Expected '{expected}' but got '{replaced.iloc[i]}'"

if __name__ == "__main__":
    test_slice_replace_consistency()
```

<details>

<summary>
**Failing input**: `strings=['0'], start=1, stop=0, repl=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 17, in <module>
    test_slice_replace_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 6, in test_slice_replace_consistency
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 14, in test_slice_replace_consistency
    assert replaced.iloc[i] == expected, f"Failed for string='{s.iloc[i]}', start={start}, stop={stop}, repl='{repl}'. Expected '{expected}' but got '{replaced.iloc[i]}'"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Failed for string='0', start=1, stop=0, repl=''. Expected '00' but got '0'
Falsifying example: test_slice_replace_consistency(
    # The test sometimes passed when commented parts were varied together.
    strings=['0'],  # or any other generated value
    start=1,
    stop=0,
    repl='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Test case that crashes
s = pd.Series(['hello'])
result = s.str.slice_replace(start=1, stop=0, repl='X')

print(f"Input string: {s.iloc[0]}")
print(f"start=1, stop=0, repl='X'")
print(f"Result: {result.iloc[0]}")
print(f"Expected (s[:start] + repl + s[stop:]): {s.iloc[0][:1] + 'X' + s.iloc[0][0:]}")

# Verify the bug
expected = s.iloc[0][:1] + 'X' + s.iloc[0][0:]
actual = result.iloc[0]

if actual == expected:
    print("\nPASSED: Result matches expected behavior")
else:
    print(f"\nFAILED: Bug confirmed!")
    print(f"  Expected: '{expected}'")
    print(f"  Got:      '{actual}'")
```

<details>

<summary>
Bug confirmed - incorrect slice replacement when start > stop
</summary>
```
Input string: hello
start=1, stop=0, repl='X'
Result: hXello
Expected (s[:start] + repl + s[stop:]): hXhello

FAILED: Bug confirmed!
  Expected: 'hXhello'
  Got:      'hXello'
```
</details>

## Why This Is A Bug

The pandas documentation for `slice_replace` states that it "Replace a positional slice of a string with another value" where "the slice from start to stop is replaced with repl". This clearly implies the operation should follow Python's standard slicing semantics: `result = s[:start] + repl + s[stop:]`.

However, the implementation contains special-case logic in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/strings/object_array.py` (lines 352-355):

```python
if x[start:stop] == "":
    local_stop = start
else:
    local_stop = stop
```

When `start >= stop`, the slice `x[start:stop]` returns an empty string (standard Python behavior), which triggers this special case. This changes `local_stop` to equal `start`, causing the function to use `x[start:]` instead of `x[stop:]` for the remainder of the string. This effectively makes the function behave as `x[:start] + repl + x[start:]` instead of the documented `x[:start] + repl + x[stop:]`.

This violates:
1. **The API documentation** - which explicitly says "the slice from start to stop is replaced"
2. **Python's standard slicing semantics** - where `s[start:stop]` is well-defined for any values of start and stop
3. **User expectations** - the method name "slice_replace" strongly suggests standard slicing behavior
4. **Consistency** - the function behaves differently for `start >= stop` versus `start < stop` cases

## Relevant Context

The bug manifests in several scenarios:
- When `start > stop`: The characters between `stop` and `start` are lost instead of being duplicated
- When `start == stop`: Works correctly only by coincidence (since `x[start:]` equals `x[stop:]` when they're equal)
- Example: `'hello'[1:0]` gives `''`, but replacing that empty slice with 'X' should yield `'hXhello'` not `'hXello'`

The pandas string methods are meant to provide vectorized string operations that mirror Python's built-in string operations. This deviation breaks that contract.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.slice_replace.html

## Proposed Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -349,16 +349,11 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
             repl = ""

         def f(x):
-            if x[start:stop] == "":
-                local_stop = start
-            else:
-                local_stop = stop
             y = ""
             if start is not None:
                 y += x[:start]
             y += repl
             if stop is not None:
-                y += x[local_stop:]
+                y += x[stop:]
             return y

         return self._str_map(f)
```