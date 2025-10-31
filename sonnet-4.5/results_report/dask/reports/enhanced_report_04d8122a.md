# Bug Report: dask.dataframe.io.sorted_division_locations Fails on Python Lists Despite Documented Examples

**Target**: `dask.dataframe.io.io.sorted_division_locations`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `sorted_division_locations` function crashes with `TypeError: No dispatch for <class 'list'>` when passed Python lists, even though all 5 examples in its docstring use Python lists as input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_sorted_division_locations_with_lists(seq, chunksize):
    """Test that sorted_division_locations works with Python lists as documented."""
    seq_sorted = sorted(seq)
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
    assert divisions[0] == seq_sorted[0]
    assert divisions[-1] == seq_sorted[-1]

# Run the test
if __name__ == "__main__":
    test_sorted_division_locations_with_lists()
```

<details>

<summary>
**Failing input**: `seq=[0], chunksize=1`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/63
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_sorted_division_locations_with_lists FAILED                [100%]

=================================== FAILURES ===================================
__________________ test_sorted_division_locations_with_lists ___________________
hypo.py:5: in test_sorted_division_locations_with_lists
    seq=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100),
               ^^^
hypo.py:11: in test_sorted_division_locations_with_lists
    divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/io.py:284: in sorted_division_locations
    seq = tolist(seq)
          ^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dispatch.py:91: in tolist
    func = tolist_dispatch.dispatch(type(obj))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:774: in dispatch
    raise TypeError(f"No dispatch for {cls}")
E   TypeError: No dispatch for <class 'list'>
E   Falsifying example: test_sorted_division_locations_with_lists(
E       seq=[0],
E       chunksize=1,
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_sorted_division_locations_with_lists - TypeError: No dis...
============================== 1 failed in 0.70s ===============================
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.io import sorted_division_locations

# Test the exact example from the function's docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
print("Testing docstring example: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=2")
try:
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with integers
print("Testing with integer list: [1, 2, 3, 4] with chunksize=2")
try:
    result = sorted_division_locations([1, 2, 3, 4], chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test minimal case
print("Testing minimal case: [0] with chunksize=1")
try:
    result = sorted_division_locations([0], chunksize=1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError: No dispatch for <class 'list'> on all test cases
</summary>
```
Testing docstring example: L = ['A', 'B', 'C', 'D', 'E', 'F'] with chunksize=2
Error: TypeError: No dispatch for <class 'list'>

==================================================

Testing with integer list: [1, 2, 3, 4] with chunksize=2
Error: TypeError: No dispatch for <class 'list'>

==================================================

Testing minimal case: [0] with chunksize=1
Error: TypeError: No dispatch for <class 'list'>
```
</details>

## Why This Is A Bug

This is a clear contract violation between the function's documentation and its implementation. The function's docstring (lines 259-278 in `/dask/dataframe/io/io.py`) contains five explicit examples, all using Python lists as input:

1. `L = ['A', 'B', 'C', 'D', 'E', 'F']` with `chunksize=2` (expected output: `(['A', 'C', 'E', 'F'], [0, 2, 4, 6])`)
2. Same list with `chunksize=3`
3. `L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']` with various chunksizes
4. `['A']` with `chunksize=2`

However, the implementation at line 284 unconditionally calls `seq = tolist(seq)` on all inputs. The `tolist` function uses a type dispatcher (`tolist_dispatch`) that only has handlers registered for:
- `np.ndarray`
- `pd.Series`
- `pd.Index`
- `pd.Categorical`
- `cupy.ndarray` (optional)

No handler is registered for Python's built-in `list` type, causing the dispatcher to raise `TypeError: No dispatch for <class 'list'>`.

The comment at lines 282-283 states: "Convert from an ndarray to a plain list so that any divisions we extract from seq are plain Python scalars." This indicates the function expects to work with lists internally, and the unconditional `tolist()` call appears to be a coding error where the developer assumed all inputs would be numpy/pandas objects.

## Relevant Context

- The function is part of the public API in `dask.dataframe.io.io`
- The function name explicitly mentions "sorted list" not "sorted array"
- All docstring examples are formatted as executable Python code with `>>>` prompts
- The `tolist_dispatch` registry is defined in `/dask/dataframe/dispatch.py`
- Handler registrations are in `/dask/dataframe/backends.py`
- The error occurs immediately on any list input, making the function unusable for its documented use cases

## Proposed Fix

```diff
--- a/dask/dataframe/backends.py
+++ b/dask/dataframe/backends.py
@@ -773,6 +773,11 @@ def is_float_na_dtype_numpy_or_pandas(obj):
     return np.issubdtype(obj, np.floating)


+@tolist_dispatch.register(list)
+def tolist_list(obj):
+    return obj
+
+
 @tolist_dispatch.register((np.ndarray, pd.Series, pd.Index, pd.Categorical))
 def tolist_numpy_or_pandas(obj):
     return obj.tolist()
```