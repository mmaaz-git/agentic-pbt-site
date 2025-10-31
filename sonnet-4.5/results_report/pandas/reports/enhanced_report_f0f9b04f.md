# Bug Report: pandas.core.indexers.length_of_indexer Returns Negative Lengths for Negative Step Slices

**Target**: `pandas.core.indexers.length_of_indexer`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `length_of_indexer` function incorrectly returns negative lengths for slices with negative steps when `start` and/or `stop` are `None`. This violates the mathematical definition of length as a non-negative value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
    n=st.integers(min_value=1, max_value=50)
)
def test_length_of_indexer_slice_comprehensive(start, stop, step, n):
    slc = slice(start, stop, step)
    target = list(range(n))

    computed_length = indexers.length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"slice({start}, {stop}, {step}) on list of length {n}: " \
        f"length_of_indexer returned {computed_length}, actual length is {actual_length}"

# Run the test
if __name__ == "__main__":
    try:
        test_length_of_indexer_slice_comprehensive()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: `slice(None, None, -1)` on list of length 1
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/envs/pandas_env
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_length_of_indexer_slice_comprehensive FAILED               [100%]

=================================== FAILURES ===================================
__________________ test_length_of_indexer_slice_comprehensive __________________

    @given(
>       start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
                   ^^^
        stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
        step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
        n=st.integers(min_value=1, max_value=50)
    )

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

start = None, stop = None, step = -1, n = 1

    @given(
        start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
        stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
        step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
        n=st.integers(min_value=1, max_value=50)
    )
    def test_length_of_indexer_slice_comprehensive(start, stop, step, n):
        slc = slice(start, stop, step)
        target = list(range(n))

        computed_length = indexers.length_of_indexer(slc, target)
        actual_length = len(target[slc])

>       assert computed_length == actual_length, \
            f"slice({start}, {stop}, {step}) on list of length {n}: " \
            f"length_of_indexer returned {computed_length}, actual length is {actual_length}"
E       AssertionError: slice(None, None, -1) on list of length 1: length_of_indexer returned -1, actual length is 1
E       assert -1 == 1
E       Falsifying example: test_length_of_indexer_slice_comprehensive(
E           start=None,
E           stop=None,
E           step=-1,
E           n=1,
```
</details>

## Reproducing the Bug

```python
import pandas.core.indexers as indexers

target = [0, 1, 2, 3, 4]
slc = slice(None, None, -1)

computed = indexers.length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Computed length: {computed}")
print(f"Actual length: {actual}")
print(f"Actual result: {target[slc]}")

# Additional test cases to demonstrate the bug
test_cases = [
    (slice(None, None, -1), [0, 1, 2, 3, 4]),
    (slice(None, None, -2), [0, 1, 2, 3, 4]),
    (slice(4, None, -1), [0, 1, 2, 3, 4]),
    (slice(None, 0, -1), [0, 1, 2, 3, 4]),
    (slice(4, 0, -1), [0, 1, 2, 3, 4]),  # This one works
]

print("\nAdditional test cases:")
for slc, target in test_cases:
    computed = indexers.length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"slice{slc.start, slc.stop, slc.step}: computed={computed}, actual={actual}, matches={computed == actual}")
```

<details>

<summary>
Output showing negative length calculations
</summary>
```
Computed length: -5
Actual length: 5
Actual result: [4, 3, 2, 1, 0]

Additional test cases:
slice(None, None, -1): computed=-5, actual=5, matches=False
slice(None, None, -2): computed=-2, actual=3, matches=False
slice(4, None, -1): computed=-1, actual=5, matches=False
slice(None, 0, -1): computed=0, actual=4, matches=False
slice(4, 0, -1): computed=4, actual=4, matches=True
```
</details>

## Why This Is A Bug

The function `length_of_indexer` is designed to return the expected length of `target[indexer]`. By mathematical definition, length is always a non-negative integer. Returning negative values like -5 or -1 violates this fundamental property.

The bug occurs due to incorrect handling of Python's slice semantics for negative steps. When a slice has a negative step and `None` for start/stop, Python uses different default values than for positive steps:

- For positive steps: `start` defaults to 0, `stop` defaults to `len(sequence)`
- For negative steps: `start` defaults to `len(sequence)-1`, `stop` defaults to -1 (which means "before index 0")

The current implementation in pandas incorrectly assumes the same defaults (start=0, stop=len) regardless of step sign, then attempts to compensate with a swap operation (`start, stop = stop + 1, start + 1`) that produces mathematically invalid negative lengths.

The function's docstring states it should "Return the expected length of target[indexer]", which clearly implies returning the same value as `len(target[indexer])` for slice objects. The current behavior fails this contract.

## Relevant Context

The `length_of_indexer` function is located in `/pandas/core/indexers/utils.py` (lines 290-329) and is used internally by pandas in several places including:
- `pandas/core/indexing.py` for indexing operations
- `pandas/core/indexers/utils.py` for validation of indexer values

While this is an internal utility function not part of the public API, it's still used in core pandas operations. The fact that pandas Series and DataFrame slicing with negative steps works correctly suggests there may be workarounds elsewhere in the codebase, but the function itself contains a clear logic error.

Python's built-in `slice.indices(length)` method correctly normalizes slice parameters for both positive and negative steps, handling all edge cases properly. This method has been part of Python since at least Python 2.3 and is the standard way to convert slice objects to normalized start, stop, step values.

Documentation reference: https://docs.python.org/3/reference/datamodel.html#slice.indices

## Proposed Fix

```diff
--- a/pandas/core/indexers/utils.py
+++ b/pandas/core/indexers/utils.py
@@ -297,22 +297,7 @@ def length_of_indexer(indexer, target=None) -> int:
     """
     if target is not None and isinstance(indexer, slice):
         target_len = len(target)
-        start = indexer.start
-        stop = indexer.stop
-        step = indexer.step
-        if start is None:
-            start = 0
-        elif start < 0:
-            start += target_len
-        if stop is None or stop > target_len:
-            stop = target_len
-        elif stop < 0:
-            stop += target_len
-        if step is None:
-            step = 1
-        elif step < 0:
-            start, stop = stop + 1, start + 1
-            step = -step
-        return (stop - start + step - 1) // step
+        start, stop, step = indexer.indices(target_len)
+        return len(range(start, stop, step))
     elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
         if isinstance(indexer, list):
```