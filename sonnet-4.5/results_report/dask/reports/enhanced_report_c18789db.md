# Bug Report: dask.sizeof Non-Deterministic Size Calculation for Collections

**Target**: `dask.sizeof`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sizeof` function in `dask.sizeof` returns different values for the same object when called repeatedly on collections (lists, tuples, dicts) with more than 10 items, violating the fundamental expectation that object size measurements should be deterministic.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st

from dask.sizeof import sizeof


@given(st.dictionaries(st.integers(), st.integers(), min_size=11, max_size=30))
@settings(max_examples=200)
def test_sizeof_dict_consistency(d):
    result1 = sizeof(d)
    result2 = sizeof(d)
    assert result1 == result2, f"sizeof(dict) should be deterministic, but got {result1} != {result2}"


if __name__ == "__main__":
    test_sizeof_dict_consistency()
```

<details>

<summary>
**Failing input**: `{0: 0, -1: 0, 2: 0, -2: 0, 3: 0, -3: 1_073_741_824, 4: 0, -4: 0, 5: 0, -5: 0, 1: 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 15, in <module>
    test_sizeof_dict_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 7, in test_sizeof_dict_consistency
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 11, in test_sizeof_dict_consistency
    assert result1 == result2, f"sizeof(dict) should be deterministic, but got {result1} != {result2}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: sizeof(dict) should be deterministic, but got 1440 != 1444
Falsifying example: test_sizeof_dict_consistency(
    d={0: 0,
     -1: 0,
     2: 0,
     -2: 0,
     3: 0,
     -3: 1_073_741_824,
     4: 0,
     -4: 0,
     5: 0,
     -5: 0,
     1: 0},
)
```
</details>

## Reproducing the Bug

```python
from dask.sizeof import sizeof

# Create a list with items of different sizes to make the non-determinism obvious
# Using strings of different lengths
lst = ["a" * i for i in range(1, 21)]  # 20 strings of lengths 1 to 20

print("Testing sizeof on a list with 20 strings of varying lengths:")
print(f"List contents: ['a', 'aa', 'aaa', ..., 'a'*20]")
print()

results = []
for i in range(10):
    result = sizeof(lst)
    results.append(result)
    print(f"Call {i+1}: {result}")

print(f"\nUnique values: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")

# Also test with a dict
d = {i: "x" * i for i in range(1, 21)}  # Dict with values of different sizes

print("\n\nTesting sizeof on a dict with 20 entries having values of varying lengths:")
print(f"Dict contents: {{1: 'x', 2: 'xx', 3: 'xxx', ..., 20: 'x'*20}}")
print()

results = []
for i in range(10):
    result = sizeof(d)
    results.append(result)
    print(f"Call {i+1}: {result}")

print(f"\nUnique values: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")
```

<details>

<summary>
Non-deterministic sizeof results varying across multiple calls
</summary>
```
Testing sizeof on a list with 20 strings of varying lengths:
List contents: ['a', 'aa', 'aaa', ..., 'a'*20]

Call 1: 1268
Call 2: 1286
Call 3: 1248
Call 4: 1310
Call 5: 1300
Call 6: 1268
Call 7: 1266
Call 8: 1264
Call 9: 1300
Call 10: 1234

Unique values: [1234, 1248, 1264, 1266, 1268, 1286, 1300, 1310]
Number of unique values: 8
Deterministic? False


Testing sizeof on a dict with 20 entries having values of varying lengths:
Dict contents: {1: 'x', 2: 'xx', 3: 'xxx', ..., 20: 'x'*20}

Call 1: 2482
Call 2: 2548
Call 3: 2494
Call 4: 2538
Call 5: 2554
Call 6: 2478
Call 7: 2470
Call 8: 2506
Call 9: 2574
Call 10: 2554

Unique values: [2470, 2478, 2482, 2494, 2506, 2538, 2548, 2554, 2574]
Number of unique values: 9
Deterministic? False
```
</details>

## Why This Is A Bug

This violates the fundamental contract of a size measurement function. The `sizeof` function should return a consistent value for the same object, as the memory footprint of an object doesn't change between calls. The non-determinism breaks several important use cases:

1. **Memory Management**: Dask uses `sizeof` for memory spilling decisions. Non-deterministic values could cause inconsistent decisions about when to spill data to disk.

2. **Task Deduplication**: If sizeof values are used in hash calculations or equality checks for deduplication, the non-determinism could cause cache misses or incorrect duplicate detection.

3. **Reproducibility**: Scientific computing workflows expect deterministic behavior for debugging and result validation. Random variations in size calculations make it impossible to verify consistent behavior across runs.

4. **API Contract**: The function name `sizeof` inherently implies measuring an intrinsic property of the object. Users expect `sizeof(x) == sizeof(x)` to always be true, similar to `len(x) == len(x)`.

## Relevant Context

The root cause is in `/envs/dask_env/lib/python3.13/site-packages/dask/sizeof.py` lines 44-59. When collections have more than 10 items, the function uses `random.sample()` to estimate the total size:

```python
@sizeof.register(list)
@sizeof.register(tuple)
@sizeof.register(set)
@sizeof.register(frozenset)
def sizeof_python_collection(seq):
    num_items = len(seq)
    num_samples = 10
    if num_items > num_samples:
        if isinstance(seq, (set, frozenset)):
            samples = itertools.islice(seq, num_samples)
        else:
            samples = random.sample(seq, num_samples)  # <-- Non-deterministic sampling
        return sys.getsizeof(seq) + int(
            num_items / num_samples * sum(map(sizeof, samples))
        )
    else:
        return sys.getsizeof(seq) + sum(map(sizeof, seq))
```

The `random.sample()` call uses Python's global random state without a fixed seed, causing different samples to be selected on each invocation. This is particularly problematic when collection items have different sizes (like strings of varying lengths or large integers mixed with small ones).

Interestingly, sets and frozensets already use deterministic sampling via `itertools.islice`, showing that the developers recognized the need for consistent sampling in some cases.

## Proposed Fix

Replace the non-deterministic `random.sample()` with deterministic sampling that maintains the performance optimization while ensuring consistent results:

```diff
--- a/sizeof.py
+++ b/sizeof.py
@@ -51,7 +51,9 @@ def sizeof_python_collection(seq):
             # the first `num_samples` items.
             samples = itertools.islice(seq, num_samples)
         else:
-            samples = random.sample(seq, num_samples)
+            # Use deterministic sampling: evenly spaced indices
+            indices = range(0, num_items, max(1, num_items // num_samples))[:num_samples]
+            samples = [seq[i] for i in indices]
         return sys.getsizeof(seq) + int(
             num_items / num_samples * sum(map(sizeof, samples))
         )
```

This fix samples elements at evenly-spaced indices, providing a representative sample without randomness. Alternative approaches could include using a seeded random generator based on the object's id: `random.Random(hash(id(seq))).sample(seq, num_samples)`, which would be deterministic per object while still using random sampling.