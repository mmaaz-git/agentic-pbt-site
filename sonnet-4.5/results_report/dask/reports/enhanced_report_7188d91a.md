# Bug Report: dask.bag.from_sequence Incorrect Partition Count When npartitions Specified

**Target**: `dask.bag.from_sequence`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `from_sequence` function does not create the requested number of partitions when the `npartitions` parameter is specified, violating the expected API contract and causing downstream failures in operations that require matching partition counts.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.bag as db


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=20),
       st.integers(min_value=1, max_value=10))
def test_from_sequence_respects_npartitions(seq, npartitions):
    bag = db.from_sequence(seq, npartitions=npartitions)

    expected = npartitions if len(seq) >= npartitions else len(seq)

    assert bag.npartitions == expected, \
        f"from_sequence(seq of len {len(seq)}, npartitions={npartitions}) " \
        f"produced {bag.npartitions} partitions, expected {expected}"


if __name__ == "__main__":
    test_from_sequence_respects_npartitions()
```

<details>

<summary>
**Failing input**: `seq=[0, 0, 0, 0, 0], npartitions=4`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 19, in <module>
    test_from_sequence_respects_npartitions()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_from_sequence_respects_npartitions
    @given(st.lists(st.integers(), min_size=1, max_size=20),
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 13, in test_from_sequence_respects_npartitions
    assert bag.npartitions == expected, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: from_sequence(seq of len 5, npartitions=4) produced 3 partitions, expected 4
Falsifying example: test_from_sequence_respects_npartitions(
    seq=[0, 0, 0, 0, 0],
    npartitions=4,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/59/hypo.py:14
```
</details>

## Reproducing the Bug

```python
import dask.bag as db

# Test case 1: Basic case from the bug report
seq = [1, 2, 3, 4]
requested_npartitions = 3

bag = db.from_sequence(seq, npartitions=requested_npartitions)

print(f"Test 1: Basic case")
print(f"Sequence: {seq}")
print(f"Requested npartitions: {requested_npartitions}")
print(f"Actual npartitions: {bag.npartitions}")
print(f"Expected: 3, Got: {bag.npartitions}")
print()

# Test case 2: Specific failing case from bug report
seq2 = [0, 0, 0, 0, 0]
requested_npartitions2 = 4

bag2 = db.from_sequence(seq2, npartitions=requested_npartitions2)

print(f"Test 2: Specific failing case")
print(f"Sequence: {seq2}")
print(f"Requested npartitions: {requested_npartitions2}")
print(f"Actual npartitions: {bag2.npartitions}")
print(f"Expected: 4, Got: {bag2.npartitions}")
print()

# Test case 3: Downstream zip() problem
print("Test 3: Downstream zip() problem")
bag1 = db.from_sequence([0, 0, 0], npartitions=3)
bag2 = db.from_sequence([0, 0, 0, 0], npartitions=3)
print(f"bag1 ([0, 0, 0], npartitions=3): {bag1.npartitions} partitions")
print(f"bag2 ([0, 0, 0, 0], npartitions=3): {bag2.npartitions} partitions")

try:
    result = db.zip(bag1, bag2)
    print("zip() succeeded")
except AssertionError as e:
    print(f"zip() failed with AssertionError: {e}")

# Final assertion to demonstrate the bug
assert bag.npartitions == requested_npartitions, f"Expected {requested_npartitions} partitions, got {bag.npartitions}"
```

<details>

<summary>
AssertionError: Expected 3 partitions, got 2
</summary>
```
Test 1: Basic case
Sequence: [1, 2, 3, 4]
Requested npartitions: 3
Actual npartitions: 2
Expected: 3, Got: 2

Test 2: Specific failing case
Sequence: [0, 0, 0, 0, 0]
Requested npartitions: 4
Actual npartitions: 3
Expected: 4, Got: 3

Test 3: Downstream zip() problem
bag1 ([0, 0, 0], npartitions=3): 3 partitions
bag2 ([0, 0, 0, 0], npartitions=3): 2 partitions
zip() failed with AssertionError:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/repo.py", line 43, in <module>
    assert bag.npartitions == requested_npartitions, f"Expected {requested_npartitions} partitions, got {bag.npartitions}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 3 partitions, got 2
```
</details>

## Why This Is A Bug

The `npartitions` parameter is documented as "The number of desired partitions" in the function docstring (line 1775 of /home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py). While the word "desired" could be interpreted as a hint, users reasonably expect this parameter to control the actual number of partitions created, especially given:

1. **Parameter naming convention**: The parameter is named `npartitions` (not `target_npartitions` or `approximate_npartitions`), implying direct control over the partition count.

2. **Inconsistent calculation logic**: The current implementation (lines 1790-1794) calculates a fixed `partition_size` using `math.ceil(len(seq) / npartitions)` for small sequences and `math.floor(len(seq) / npartitions)` for larger ones, then uses `partition_all()` to create partitions. This approach cannot guarantee the exact number of partitions when the sequence length is not evenly divisible by the requested partition count.

3. **Downstream API requirements**: Functions like `dask.bag.zip()` require all input bags to have the same number of partitions (enforced by an assertion). This makes it impossible to zip bags created from sequences of different lengths even when the same `npartitions` value is specified, as demonstrated in the reproduction code.

4. **Principle of least surprise**: When a user specifies `npartitions=3`, they expect 3 partitions, not 2. The current behavior violates this expectation without any warning in the documentation.

## Relevant Context

- The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/dask/bag/core.py` lines 1789-1808
- The issue stems from using a fixed partition size calculation followed by `partition_all()` which doesn't guarantee exact partition counts
- Other dask functions like `bag_range` do guarantee exact partition counts with their `npartitions` parameter, creating an inconsistency in the API
- Documentation: https://docs.dask.org/en/stable/bag-api.html#dask.bag.from_sequence
- Source code: https://github.com/dask/dask/blob/main/dask/bag/core.py#L1761

## Proposed Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -1789,19 +1789,31 @@ def from_sequence(seq, partition_size=None, npartitions=None):
     seq = list(seq)
     if npartitions and not partition_size:
-        if len(seq) <= 100:
-            partition_size = int(math.ceil(len(seq) / npartitions))
-        else:
-            partition_size = max(1, int(math.floor(len(seq) / npartitions)))
+        # Ensure we create exactly npartitions (or len(seq) if smaller)
+        npartitions = min(npartitions, len(seq)) if len(seq) > 0 else 1
+        base_size = len(seq) // npartitions
+        remainder = len(seq) % npartitions
+        parts = []
+        start = 0
+        for i in range(npartitions):
+            size = base_size + (1 if i < remainder else 0)
+            parts.append(seq[start:start + size])
+            start += size
+        name = "from_sequence-" + tokenize(seq, npartitions)
+        d = {(name, i): list(part) for i, part in enumerate(parts)}
+        return Bag(d, name, len(d))
+
     if npartitions is None and partition_size is None:
         if len(seq) <= 100:
             partition_size = 1
         else:
             partition_size = max(1, math.ceil(math.sqrt(len(seq)) / math.sqrt(100)))

     parts = list(partition_all(partition_size, seq))
     name = "from_sequence-" + tokenize(seq, partition_size)
     if len(parts) > 0:
         d = {(name, i): list(part) for i, part in enumerate(parts)}
     else:
         d = {(name, 0): []}

     return Bag(d, name, len(d))
```