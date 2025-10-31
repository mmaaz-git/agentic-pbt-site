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
```

**Failing inputs**:
- `head=[['', '']]`, `tail=[['']]` - Expected 2 elements in result, got 1
- `head=[['']]`, `tail=[['', '']]` - Expected 2 elements in result, got 1

## Reproducing the Bug

```python
from pandas.io.formats.printing import _justify

head = [['a', 'b', 'c']]
tail = [['x']]
result_head, result_tail = _justify(head, tail)

print(f"Input:  head={head}, tail={tail}")
print(f"Output: head={result_head}, tail={result_tail}")
print(f"Expected: head should have 3 elements, tail should have 1")
print(f"Actual:   head has {len(result_head[0])} elements (lost 2!)")
```

Output:
```
Input:  head=[['a', 'b', 'c']], tail=[['x']]
Output: head=[('a',)], tail=[('x',)]
Expected: head should have 3 elements, tail should have 1
Actual:   head has 1 elements (lost 2!)
```

## Why This Is A Bug

The function silently drops data when sequences have different lengths. This violates the invariant that justification should only add padding (spaces), not remove content. The function's type signature and docstring don't indicate that all sequences must have the same length, so this behavior is unexpected and dangerous.

The root cause is on line 484 of printing.py:

```python
max_length = [0] * len(combined[0])
for inner_seq in combined:
    length = [len(item) for item in inner_seq]
    max_length = [max(x, y) for x, y in zip(max_length, length)]  # zip truncates!
```

The `zip` truncates to the shortest sequence, so if `combined[0]` has 3 elements but `combined[1]` has 1, only the first element's max length is computed.

## Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -480,9 +480,15 @@ def _justify(
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