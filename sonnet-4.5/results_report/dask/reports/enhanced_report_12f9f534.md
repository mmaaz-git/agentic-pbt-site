# Bug Report: dask.sizeof dict Non-Determinism

**Target**: `dask.sizeof.sizeof` (dict implementation)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sizeof` function for dictionaries returns different values across multiple calls with the same input when the dictionary contains more than 10 items, violating the expected deterministic behavior of a size measurement function.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis property-based test that reveals the dask.sizeof dict non-determinism bug.
Tests that sizeof for dictionaries should follow a predictable formula.
"""
from hypothesis import given, settings, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.sizeof import sizeof

@given(st.dictionaries(st.text(), st.integers()))
@settings(max_examples=500)
def test_sizeof_dict_formula(d):
    expected = (
        sys.getsizeof(d)
        + sizeof(list(d.keys()))
        + sizeof(list(d.values()))
        - 2 * sizeof(list())
    )
    assert sizeof(d) == expected

if __name__ == "__main__":
    test_sizeof_dict_formula()
```

<details>

<summary>
**Failing input**: `d={'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '00': 0, '000': 0, '0000': 0, '00000': 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 23, in <module>
    test_sizeof_dict_formula()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 12, in test_sizeof_dict_formula
    @settings(max_examples=500)
                  ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 20, in test_sizeof_dict_formula
    assert sizeof(d) == expected
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_sizeof_dict_formula(
    d={'': 0,
     '0': 0,
     '1': 0,
     '2': 0,
     '3': 0,
     '4': 0,
     '5': 0,
     '00': 0,
     '000': 0,
     '0000': 0,
     '00000': 0},
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the dask.sizeof dict non-determinism bug.
Demonstrates that sizeof(dict) returns different values on repeated calls
when the dict has more than 10 items.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.sizeof import sizeof

# Create a dictionary with 11 items (triggers the bug)
d = {'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '00': 0, '000': 0, '0000': 0, '00000': 0, '000000': 0}

# Call sizeof multiple times on the same dict
results = [sizeof(d) for _ in range(20)]

print(f"Dictionary: {d}")
print(f"Number of items: {len(d)}")
print(f"\nResults from 20 calls to sizeof(d):")
print(results)
print(f"\nUnique values: {sorted(set(results))}")
print(f"Min value: {min(results)}")
print(f"Max value: {max(results)}")
print(f"Range: {max(results) - min(results)} bytes difference")

# Show that the issue is due to > 10 items
print("\n--- Comparison with 10 items (should be deterministic) ---")
d_10 = {str(i): i for i in range(10)}
results_10 = [sizeof(d_10) for _ in range(20)]
print(f"Dict with 10 items - Results: {sorted(set(results_10))}")
print(f"Is deterministic: {len(set(results_10)) == 1}")

print("\n--- Comparison with 11 items (non-deterministic) ---")
d_11 = {str(i): i for i in range(11)}
results_11 = [sizeof(d_11) for _ in range(20)]
print(f"Dict with 11 items - Results: {sorted(set(results_11))}")
print(f"Is deterministic: {len(set(results_11)) == 1}")
```

<details>

<summary>
Non-deterministic sizeof results varying by up to 7 bytes
</summary>
```
Dictionary: {'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '00': 0, '000': 0, '0000': 0, '00000': 0, '000000': 0}
Number of items: 11

Results from 20 calls to sizeof(d):
[1441, 1441, 1435, 1439, 1435, 1438, 1437, 1441, 1441, 1441, 1439, 1441, 1441, 1440, 1441, 1441, 1442, 1439, 1440, 1437]

Unique values: [1435, 1437, 1438, 1439, 1440, 1441, 1442]
Min value: 1435
Max value: 1442
Range: 7 bytes difference

--- Comparison with 10 items (should be deterministic) ---
Dict with 10 items - Results: [1164]
Is deterministic: True

--- Comparison with 11 items (non-deterministic) ---
Dict with 11 items - Results: [1426, 1427]
Is deterministic: False
```
</details>

## Why This Is A Bug

This violates the fundamental expectation that a size measurement function should be deterministic. Calling `sizeof(x)` multiple times with the same immutable object `x` should always return the same value. This is critical because:

1. **Consistency Expectations**: Similar Python functions like `sys.getsizeof()`, `len()`, and `.__sizeof__()` are all deterministic. Users naturally expect the same from `dask.sizeof`.

2. **Memory Management Impact**: According to Dask documentation, `sizeof` is used "to determine which objects to spill to disk" during memory management. Non-deterministic sizing can lead to:
   - Inconsistent spilling decisions for the same data
   - Unpredictable memory usage patterns
   - Difficulty debugging memory-related issues

3. **Testing and Reproducibility**: Non-deterministic functions make code harder to test, debug, and reason about, especially in distributed computing contexts where Dask operates.

The non-determinism occurs due to the interaction between two functions:
- `sizeof_python_dict` (lines 91-98) calculates dict size by summing the sizes of keys and values lists
- `sizeof_python_collection` (lines 40-59) uses `random.sample()` for collections with >10 items to estimate size
- When a dict has >10 items, its keys and values lists each trigger random sampling, introducing non-determinism

## Relevant Context

The bug manifests at a specific threshold:
- Dictionaries with ≤10 items: Always deterministic (no sampling)
- Dictionaries with >10 items: Non-deterministic due to random sampling of keys and values

The random sampling is an optimization to avoid iterating through very large collections, trading accuracy for performance. However, this optimization inadvertently makes the dict sizeof non-deterministic.

Relevant code locations:
- Dict sizeof implementation: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/sizeof.py:91-98`
- Random sampling logic: `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/sizeof.py:40-59`

The Dask documentation doesn't explicitly state that `sizeof` should be deterministic, but this is an implicit expectation for utility functions, especially those used in memory management decisions.

## Proposed Fix

```diff
--- a/dask/sizeof.py
+++ b/dask/sizeof.py
@@ -91,9 +91,20 @@ def sizeof_blocked(d):
 @sizeof.register(dict)
 def sizeof_python_dict(d):
+    # Avoid non-determinism from random sampling in sizeof_python_collection
+    # by directly calculating the size of all keys and values
+    keys_list = list(d.keys())
+    values_list = list(d.values())
+
+    # Calculate sizes without sampling to ensure determinism
+    keys_size = sys.getsizeof(keys_list) + sum(sizeof(k) for k in keys_list)
+    values_size = sys.getsizeof(values_list) + sum(sizeof(v) for v in values_list)
+    empty_list_size = sys.getsizeof(list())
+
     return (
         sys.getsizeof(d)
-        + sizeof(list(d.keys()))
-        + sizeof(list(d.values()))
-        - 2 * sizeof(list())
+        + keys_size
+        + values_size
+        - 2 * empty_list_size
     )
```