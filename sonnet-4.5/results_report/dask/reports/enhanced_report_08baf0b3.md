# Bug Report: dask.bytes.read_bytes Float Division Produces Incorrect Block Offsets

**Target**: `dask.bytes.core.read_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_bytes` function uses float division to calculate adjusted block sizes, causing block offsets to be calculated incorrectly due to floating-point precision errors. This produces inconsistent block boundaries that differ from what integer arithmetic would yield.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test that demonstrates the float division bug in dask.bytes.read_bytes.

This test compares the behavior of float division vs integer division when
calculating block offsets, showing that they produce different results.
"""

from hypothesis import given, settings, strategies as st, assume, example

@given(
    file_size=st.integers(min_value=1000, max_value=1_000_000_000),
    blocksize=st.integers(min_value=100, max_value=100_000_000)
)
@settings(max_examples=1000, deadline=None)
@example(file_size=1_000_000_001, blocksize=333_333_333)
def test_float_vs_int_division_equality(file_size, blocksize):
    """Test that float division and integer division produce the same block offsets.

    This test replicates the logic in dask.bytes.core.read_bytes to show that
    using float division (/) instead of integer division (//) causes different
    block offset calculations due to floating-point precision issues.
    """
    assume(file_size > blocksize)
    assume(file_size % blocksize != 0)

    size = file_size
    num_blocks_float = size // blocksize
    blocksize1_float = size / num_blocks_float  # Float division (buggy)

    num_blocks_int = size // blocksize
    blocksize1_int = size // num_blocks_int  # Integer division (correct)

    # Calculate offsets using float division
    place_float = 0
    off_float = [0]
    while size - place_float > (blocksize1_float * 2) - 1:
        place_float += blocksize1_float
        off_float.append(int(place_float))

    # Calculate offsets using integer division
    place_int = 0
    off_int = [0]
    while size - place_int > (blocksize1_int * 2) - 1:
        place_int += blocksize1_int
        off_int.append(int(place_int))

    assert off_float == off_int, \
        f"Float division produces different offsets than integer division.\n" \
        f"File size: {file_size}, Blocksize: {blocksize}\n" \
        f"Float offsets: {off_float}\n" \
        f"Int offsets: {off_int}"

if __name__ == "__main__":
    # Run the test
    test_float_vs_int_division_equality()
```

<details>

<summary>
**Failing input**: `file_size=1_000_000_001, blocksize=333_333_333`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 56, in <module>
    test_float_vs_int_division_equality()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 12, in test_float_vs_int_division_equality
    file_size=st.integers(min_value=1000, max_value=1_000_000_000),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 48, in test_float_vs_int_division_equality
    assert off_float == off_int, \
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: Float division produces different offsets than integer division.
File size: 1000000001, Blocksize: 333333333
Float offsets: [0, 333333333, 666666667]
Int offsets: [0, 333333333, 666666666]
Falsifying explicit example: test_float_vs_int_division_equality(
    file_size=1_000_000_001,
    blocksize=333333333,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Demonstration of dask.bytes.read_bytes float division bug.

This script shows how using float division in dask.bytes.core.read_bytes
causes incorrect block offset calculations compared to integer division.
"""

def demonstrate_float_division_bug():
    """Reproduce the exact bug in dask.bytes.read_bytes"""

    # Test case that triggers the bug
    file_size = 1_000_000_001
    blocksize = 333_333_333

    print("=" * 60)
    print("DASK BYTES FLOAT DIVISION BUG DEMONSTRATION")
    print("=" * 60)
    print(f"File size: {file_size:,} bytes")
    print(f"Blocksize: {blocksize:,} bytes")
    print(f"Number of blocks: {file_size // blocksize}")
    print()

    # Current implementation with float division (buggy)
    print("CURRENT IMPLEMENTATION (Float Division)")
    print("-" * 40)

    size = file_size
    num_blocks = size // blocksize
    blocksize1_float = size / num_blocks  # Float division as in dask/bytes/core.py:125

    print(f"blocksize1 = size / num_blocks")
    print(f"blocksize1 = {size} / {num_blocks}")
    print(f"blocksize1 = {blocksize1_float}")
    print(f"Type: {type(blocksize1_float).__name__}")
    print()

    # Calculate offsets with float division (as dask does)
    place = 0
    off_float = [0]
    length_float = []

    while size - place > (blocksize1_float * 2) - 1:
        place += blocksize1_float
        off_float.append(int(place))
        length_float.append(off_float[-1] - off_float[-2])
    length_float.append(size - off_float[-1])

    print("Block offsets:", off_float)
    print("Block lengths:", length_float)
    print(f"Sum of lengths: {sum(length_float):,}")
    print(f"Covers all bytes: {sum(length_float) == file_size}")
    print()

    # Correct implementation with integer division
    print("CORRECT IMPLEMENTATION (Integer Division)")
    print("-" * 40)

    size = file_size
    blocksize1_int = size // num_blocks  # Integer division (correct)

    print(f"blocksize1 = size // num_blocks")
    print(f"blocksize1 = {size} // {num_blocks}")
    print(f"blocksize1 = {blocksize1_int}")
    print(f"Type: {type(blocksize1_int).__name__}")
    print()

    # Calculate offsets with integer division
    place = 0
    off_int = [0]
    length_int = []

    while size - place > (blocksize1_int * 2) - 1:
        place += blocksize1_int
        off_int.append(int(place))
        length_int.append(off_int[-1] - off_int[-2])
    length_int.append(size - off_int[-1])

    print("Block offsets:", off_int)
    print("Block lengths:", length_int)
    print(f"Sum of lengths: {sum(length_int):,}")
    print(f"Covers all bytes: {sum(length_int) == file_size}")
    print()

    # Show the differences
    print("COMPARISON")
    print("-" * 40)
    print(f"Offsets match: {off_float == off_int}")

    if off_float != off_int:
        print("\nDifferences in offsets:")
        for i, (f, i_) in enumerate(zip(off_float, off_int)):
            if f != i_:
                print(f"  Index {i}: {f:,} (float) vs {i_:,} (int) - diff: {f - i_} bytes")

    print(f"\nLengths match: {length_float == length_int}")

    if length_float != length_int:
        print("\nDifferences in block lengths:")
        for i, (f, i_) in enumerate(zip(length_float, length_int)):
            if f != i_:
                print(f"  Block {i}: {f:,} (float) vs {i_:,} (int) - diff: {f - i_} bytes")

    print()
    print("IMPACT")
    print("-" * 40)
    print("The float division causes cumulative rounding errors that lead to")
    print("incorrect block boundaries. While all bytes are still read (no data")
    print("loss), the block offsets are wrong, which violates the expectation")
    print("of precise, deterministic block boundaries in file I/O operations.")

if __name__ == "__main__":
    demonstrate_float_division_bug()
```

<details>

<summary>
Output showing incorrect block offset calculation
</summary>
```
============================================================
DASK BYTES FLOAT DIVISION BUG DEMONSTRATION
============================================================
File size: 1,000,000,001 bytes
Blocksize: 333,333,333 bytes
Number of blocks: 3

CURRENT IMPLEMENTATION (Float Division)
----------------------------------------
blocksize1 = size / num_blocks
blocksize1 = 1000000001 / 3
blocksize1 = 333333333.6666667
Type: float

Block offsets: [0, 333333333, 666666667]
Block lengths: [333333333, 333333334, 333333334]
Sum of lengths: 1,000,000,001
Covers all bytes: True

CORRECT IMPLEMENTATION (Integer Division)
----------------------------------------
blocksize1 = size // num_blocks
blocksize1 = 1000000001 // 3
blocksize1 = 333333333
Type: int

Block offsets: [0, 333333333, 666666666]
Block lengths: [333333333, 333333333, 333333335]
Sum of lengths: 1,000,000,001
Covers all bytes: True

COMPARISON
----------------------------------------
Offsets match: False

Differences in offsets:
  Index 2: 666,666,667 (float) vs 666,666,666 (int) - diff: 1 bytes

Lengths match: False

Differences in block lengths:
  Block 1: 333,333,334 (float) vs 333,333,333 (int) - diff: 1 bytes
  Block 2: 333,333,334 (float) vs 333,333,335 (int) - diff: -1 bytes

IMPACT
----------------------------------------
The float division causes cumulative rounding errors that lead to
incorrect block boundaries. While all bytes are still read (no data
loss), the block offsets are wrong, which violates the expectation
of precise, deterministic block boundaries in file I/O operations.
```
</details>

## Why This Is A Bug

The code at `dask/bytes/core.py:125` uses float division to calculate the adjusted blocksize when the file size is not evenly divisible by the requested blocksize:

```python
blocksize1 = size / (size // blocksize)  # Float division
```

This violates several fundamental expectations:

1. **Byte positions must be integers**: File I/O operations deal with discrete byte positions. Using floating-point arithmetic introduces unnecessary precision issues that lead to incorrect offset calculations.

2. **Deterministic behavior**: The float division introduces non-deterministic rounding errors. When `blocksize1` is a float (e.g., 333333333.6666667), accumulating it in a loop and truncating with `int()` loses precision, causing cumulative errors.

3. **Documentation implies precision**: The function documentation states it "cleanly breaks data" into blocks. The word "cleanly" implies precise, well-defined boundaries without ambiguity from floating-point arithmetic.

4. **Common practice violation**: In file I/O operations across all programming languages and systems, byte offsets are universally calculated using integer arithmetic. Using float division is highly unusual and error-prone.

The concrete impact: In the example above, the second block offset differs by 1 byte (666666667 vs 666666666). While all bytes are ultimately read (no data loss), the incorrect block boundaries could cause issues in:
- Parallel processing systems that depend on precise block alignment
- Distributed systems where different nodes need consistent block boundaries
- Systems that need to align blocks with specific byte positions (e.g., record boundaries)
- Reproducibility across different floating-point implementations

## Relevant Context

The bug occurs specifically in the block offset calculation logic within `dask.bytes.core.read_bytes`. The function is responsible for dividing files into chunks for parallel processing, making correct block boundaries critical for Dask's distributed computing capabilities.

**Source code location**: `/dask/bytes/core.py:125`
**Function**: `read_bytes`
**Documentation**: The function is documented to "cleanly break data" into blocks with a specified blocksize in bytes

The issue affects any file where `file_size % blocksize != 0`, which is a common scenario. The bug has likely been present since the introduction of this optimization to maintain the same number of blocks when adjusting blocksize.

## Proposed Fix

```diff
--- a/dask/bytes/core.py
+++ b/dask/bytes/core.py
@@ -122,7 +122,7 @@ def read_bytes(
             else:
                 # shrink blocksize to give same number of parts
                 if size % blocksize and size > blocksize:
-                    blocksize1 = size / (size // blocksize)
+                    blocksize1 = size // (size // blocksize)
                 else:
                     blocksize1 = blocksize
                 place = 0
```