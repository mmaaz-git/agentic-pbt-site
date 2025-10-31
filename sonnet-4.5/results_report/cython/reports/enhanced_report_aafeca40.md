# Bug Report: Cython.Tempita._looper odd/even Properties Return Inverted Boolean Values

**Target**: `Cython.Tempita._looper.loop_pos`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `odd` and `even` properties of the `loop_pos` class in Cython's bundled Tempita library return mathematically incorrect values - position 0 (first item) returns odd=True when it should be even, and the even property returns an integer instead of a boolean.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1))
def test_looper_odd_even_properties(seq):
    result = list(looper(seq))
    for loop_obj, item in result:
        pos = loop_obj.index
        if pos % 2 == 0:
            assert loop_obj.even == True
            assert loop_obj.odd == False
        else:
            assert loop_obj.odd == True
            assert loop_obj.even == False


if __name__ == "__main__":
    test_looper_odd_even_properties()
```

<details>

<summary>
**Failing input**: `seq=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 19, in <module>
    test_looper_odd_even_properties()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_looper_odd_even_properties
    def test_looper_odd_even_properties(seq):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 11, in test_looper_odd_even_properties
    assert loop_obj.even == True
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_looper_odd_even_properties(
    seq=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Tempita._looper import looper

print("Testing looper odd/even properties:")
print("=" * 50)

for loop_obj, item in looper([1, 2, 3, 4]):
    pos = loop_obj.index
    print(f"Position {pos}: odd={loop_obj.odd}, even={loop_obj.even}")

    # Show what the values SHOULD be based on mathematical definitions
    expected_odd = pos % 2 == 1
    expected_even = pos % 2 == 0
    print(f"  Expected: odd={expected_odd}, even={expected_even}")

    # Check if they match
    if loop_obj.odd != expected_odd or bool(loop_obj.even) != expected_even:
        print(f"  ❌ MISMATCH!")
    else:
        print(f"  ✓ Match")
    print()

print("=" * 50)
print("\nDetailed analysis of position 0:")
loop_list = list(looper([1]))
loop_obj, item = loop_list[0]
print(f"Position: {loop_obj.index}")
print(f"odd property returns: {loop_obj.odd} (type: {type(loop_obj.odd).__name__})")
print(f"even property returns: {loop_obj.even} (type: {type(loop_obj.even).__name__})")
print(f"\nMathematically, 0 % 2 = {0 % 2}, so position 0 is EVEN")
print(f"Therefore, odd should be False and even should be True")
print(f"But we got odd={loop_obj.odd} and even={loop_obj.even}")
```

<details>

<summary>
All positions show inverted odd/even values with type inconsistency
</summary>
```
Testing looper odd/even properties:
==================================================
Position 0: odd=True, even=0
  Expected: odd=False, even=True
  ❌ MISMATCH!

Position 1: odd=False, even=1
  Expected: odd=True, even=False
  ❌ MISMATCH!

Position 2: odd=True, even=0
  Expected: odd=False, even=True
  ❌ MISMATCH!

Position 3: odd=False, even=1
  Expected: odd=True, even=False
  ❌ MISMATCH!

==================================================

Detailed analysis of position 0:
Position: 0
odd property returns: True (type: bool)
even property returns: 0 (type: int)

Mathematically, 0 % 2 = 0, so position 0 is EVEN
Therefore, odd should be False and even should be True
But we got odd=True and even=0
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Mathematical Incorrectness**: In mathematics and computer science, 0 is universally considered an even number (0 % 2 == 0). The implementation returns odd=True for position 0, which is mathematically wrong.

2. **Documentation Contradiction**: The original Tempita documentation explicitly states: "The first item is even." The current implementation makes the first item (position 0) return odd=True, directly contradicting this documented behavior.

3. **Semantic Violation**: Properties named `odd` and `even` have well-established meanings in mathematics. Users reasonably expect these to follow standard mathematical conventions.

4. **Type Inconsistency**: The `odd` property returns a boolean (True/False) while the `even` property returns an integer (0/1). This asymmetry is unexpected and undocumented.

5. **Logic Inversion**: The implementation has the boolean logic backwards:
   - `odd` uses `not self.pos % 2`, which returns True for even positions
   - `even` uses `self.pos % 2`, which returns 0 (falsy) for even positions

## Relevant Context

The bug exists in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_looper.py` at lines 98-104. This is Cython's bundled version of the Tempita template library. The original Tempita project appears to have the same bug, suggesting this issue has been present for years.

The looper utility is designed to provide context information when iterating through sequences in templates. Other properties like `first`, `last`, and `index` work correctly - only `odd` and `even` are affected.

Workaround: Users can avoid the buggy properties by using `loop_obj.index % 2 == 0` for even and `loop_obj.index % 2 == 1` for odd checks.

## Proposed Fix

```diff
--- a/Cython/Tempita/_looper.py
+++ b/Cython/Tempita/_looper.py
@@ -96,11 +96,11 @@ class loop_pos:
     previous = property(previous)

     def odd(self):
-        return not self.pos % 2
+        return self.pos % 2 == 1
     odd = property(odd)

     def even(self):
-        return self.pos % 2
+        return self.pos % 2 == 0
     even = property(even)

     def first(self):
```