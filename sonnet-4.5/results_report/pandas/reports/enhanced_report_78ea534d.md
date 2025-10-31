# Bug Report: pandas.io.formats.printing._justify silently truncates data

**Target**: `pandas.io.formats.printing._justify`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_justify` function silently drops elements when input sequences have different lengths, causing data loss without any error or warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.printing import _justify


@given(
    st.lists(st.lists(st.text(max_size=10), min_size=1, max_size=5), min_size=1, max_size=5),
    st.lists(st.lists(st.text(max_size=10), min_size=1, max_size=5), min_size=1, max_size=5)
)
def test_justify_preserves_content(head, tail):
    """
    Property: _justify should preserve all content from head and tail.
    """
    result_head, result_tail = _justify(head, tail)

    for i, (orig, justified) in enumerate(zip(head, result_head)):
        assert len(justified) == len(orig), \
            f"head[{i}] length changed: {len(orig)} -> {len(justified)}"

    for i, (orig, justified) in enumerate(zip(tail, result_tail)):
        assert len(justified) == len(orig), \
            f"tail[{i}] length changed: {len(orig)} -> {len(justified)}"


if __name__ == "__main__":
    test_justify_preserves_content()
```

<details>

<summary>
**Failing input**: `head=[['', '']]`, `tail=[['']]` and `head=[['']]`, `tail=[['', '']]`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 25, in <module>
  |     test_justify_preserves_content()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 6, in test_justify_preserves_content
  |     st.lists(st.lists(st.text(max_size=10), min_size=1, max_size=5), min_size=1, max_size=5),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 16, in test_justify_preserves_content
    |     assert len(justified) == len(orig), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: head[0] length changed: 2 -> 1
    | Falsifying example: test_justify_preserves_content(
    |     head=[['', '']],
    |     tail=[['']],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 20, in test_justify_preserves_content
    |     assert len(justified) == len(orig), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: tail[0] length changed: 2 -> 1
    | Falsifying example: test_justify_preserves_content(
    |     head=[['']],
    |     tail=[['', '']],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.printing import _justify

# Test case demonstrating the bug where _justify silently drops elements
head = [['a', 'b', 'c']]
tail = [['x']]
result_head, result_tail = _justify(head, tail)

print(f"Input:  head={head}, tail={tail}")
print(f"Output: head={result_head}, tail={result_tail}")
print(f"Expected: head should have 3 elements, tail should have 1 element")
print(f"Actual:   head has {len(result_head[0])} element(s) (lost 2!), tail has {len(result_tail[0])} element(s)")
print()
print("Data Loss: Elements 'b' and 'c' were silently dropped from head!")
```

<details>

<summary>
Data loss demonstration - elements silently dropped
</summary>
```
Input:  head=[['a', 'b', 'c']], tail=[['x']]
Output: head=[('a',)], tail=[('x',)]
Expected: head should have 3 elements, tail should have 1 element
Actual:   head has 1 element(s) (lost 2!), tail has 1 element(s)

Data Loss: Elements 'b' and 'c' were silently dropped from head!
```
</details>

## Why This Is A Bug

This bug violates multiple expected behaviors and documented contracts:

1. **Silent Data Loss**: The function silently discards user data without any error or warning. When `head` contains 3 elements and `tail` contains 1, the function only returns 1 element for each, dropping 2 elements from `head`.

2. **Documentation Contract Violation**: The docstring states the function returns "Same as head and tail, but items are right aligned when stacked vertically." The phrase "Same as head and tail" explicitly promises content preservation - only alignment should change, not the actual data.

3. **Function Name Semantics**: The function is named `_justify`, which in text formatting contexts means adjusting spacing/padding for alignment purposes. Justification operations should never remove content, only add formatting.

4. **No Length Constraint Documented**: The documentation does not specify that all sequences must have the same length. It simply states "list-like of list-likes of strings" for both parameters, implying the function should handle variable-length sequences.

5. **Incorrect Implementation Assumption**: The root cause is at line 484 of printing.py where the code assumes all sequences have the same length as the first sequence: `max_length = [0] * len(combined[0])`. When sequences have different lengths, the subsequent `zip()` operations at lines 487, 491, and 494 truncate to the shortest sequence, causing data loss.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/printing.py` at lines 484-494.

This is an internal function (indicated by the leading underscore) used by pandas for formatting output when displaying data structures. While not directly exposed to users, it's imported and used by other pandas components for display formatting. The function is tested in `pandas/tests/io/formats/test_printing.py`.

The example in the docstring (`_justify([['a', 'b']], [['abc', 'abcd']])`) only demonstrates equal-length sequences, which doesn't clarify the expected behavior for unequal lengths and may have masked this issue during development.

Documentation: https://github.com/pandas-dev/pandas/blob/main/pandas/io/formats/printing.py

## Proposed Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -480,15 +480,29 @@ def _justify(
     """
     combined = head + tail

+    if not combined:
+        return [], []
+
+    # Find the maximum length across all sequences
+    max_seq_len = max(len(inner_seq) for inner_seq in combined)
+
     # For each position for the sequences in ``combined``,
     # find the length of the largest string.
-    max_length = [0] * len(combined[0])
+    max_length = [0] * max_seq_len
     for inner_seq in combined:
         length = [len(item) for item in inner_seq]
-        max_length = [max(x, y) for x, y in zip(max_length, length)]
+        # Pad inner_seq lengths with 0 for missing elements
+        length_padded = length + [0] * (max_seq_len - len(length))
+        max_length = [max(x, y) for x, y in zip(max_length, length_padded)]

     # justify each item in each list-like in head and tail using max_length
     head_tuples = [
-        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in head
+        tuple(
+            seq[i].rjust(max_length[i]) if i < len(seq) else ' ' * max_length[i]
+            for i in range(max_seq_len)
+        )
+        for seq in head
     ]
     tail_tuples = [
-        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in tail
+        tuple(
+            seq[i].rjust(max_length[i]) if i < len(seq) else ' ' * max_length[i]
+            for i in range(max_seq_len)
+        )
+        for seq in tail
     ]
     return head_tuples, tail_tuples
```