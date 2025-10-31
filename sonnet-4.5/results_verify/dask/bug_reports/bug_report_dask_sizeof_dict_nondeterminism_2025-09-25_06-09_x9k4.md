# Bug Report: dask.sizeof dict Non-Determinism

**Target**: `dask.sizeof.sizeof` (dict implementation)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sizeof` function for dictionaries is non-deterministic when the dictionary has more than 10 items, returning different values across multiple calls with the same input.

## Property-Based Test

```python
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
```

**Failing input**: `d={'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '00': 0, '000': 0, '0000': 0, '00000': 0, '000000': 0}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.sizeof import sizeof

d = {str(i): i for i in range(11)}

results = [sizeof(d) for _ in range(20)]

print(f"Results from 20 calls: {results}")
print(f"Unique values: {sorted(set(results))}")
print(f"Range: {min(results)} to {max(results)}")
```

**Output:**
```
Results from 20 calls: [1441, 1435, 1440, 1435, ...]
Unique values: [1435, 1437, 1439, 1440, 1441, 1442]
Range: 1435 to 1442
```

## Why This Is A Bug

The `sizeof` function should be deterministic: calling `sizeof(x)` multiple times with the same `x` should always return the same value. This is a fundamental expectation for a utility function.

The bug occurs because:

1. `sizeof_python_dict` (lines 91-98) computes the size as:
   ```python
   sys.getsizeof(d) + sizeof(list(d.keys())) + sizeof(list(d.values())) - 2 * sizeof(list())
   ```

2. `sizeof_python_collection` (lines 40-59) uses random sampling for lists with more than 10 items:
   ```python
   if num_items > num_samples:
       samples = random.sample(seq, num_samples)
       return sys.getsizeof(seq) + int(num_items / num_samples * sum(map(sizeof, samples)))
   ```

3. When a dict has >10 items, `list(d.keys())` and `list(d.values())` each have >10 items, triggering random sampling

4. Random sampling introduces non-determinism, causing `sizeof(d)` to vary across calls

This violates the basic contract of a size estimation function and can cause issues in code that relies on consistent size estimates for caching, memory management, or decision-making.

## Fix

The issue is that `sizeof_python_dict` doesn't account for the fact that `sizeof(list(...))` uses random sampling. There are two potential fixes:

**Option 1:** Make dict sizeof bypass the random sampling for its internal key/value lists:

```diff
diff --git a/dask/sizeof.py b/dask/sizeof.py
index 1234567..abcdefg 100644
--- a/dask/sizeof.py
+++ b/dask/sizeof.py
@@ -90,9 +90,15 @@ def sizeof_blocked(d):

 @sizeof.register(dict)
 def sizeof_python_dict(d):
+    # Calculate size of all keys and values to avoid non-determinism
+    # from random sampling in sizeof_python_collection
+    keys_size = sys.getsizeof(list(d.keys())) + sum(sizeof(k) for k in d.keys())
+    values_size = sys.getsizeof(list(d.values())) + sum(sizeof(v) for v in d.values())
     return (
         sys.getsizeof(d)
-        + sizeof(list(d.keys()))
-        + sizeof(list(d.values()))
+        + keys_size
+        + values_size
         - 2 * sizeof(list())
     )
```

**Option 2:** Add a flag to disable random sampling for internal calls, ensuring determinism.

Option 1 is simpler and more direct, though it may be slower for very large dicts. However, it preserves the fundamental property that `sizeof` should be deterministic.