# Bug Report: xarray.core.formatting maybe_truncate and pretty_print violate length constraints

**Target**: `xarray.core.formatting.maybe_truncate` and `xarray.core.formatting.pretty_print`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The functions `maybe_truncate()` and `pretty_print()` violate their length constraints when given small maxlen/numchars values (< 3), returning strings longer than specified limits.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.core.formatting import maybe_truncate, pretty_print

@given(st.text(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=100)
def test_maybe_truncate_at_most_maxlen(text, maxlen):
    result = maybe_truncate(text, maxlen)
    assert len(result) <= maxlen, f"maybe_truncate({text!r}, maxlen={maxlen}) returned {result!r} with length {len(result)}, expected <= {maxlen}"

@given(st.text(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=100)
def test_pretty_print_produces_exact_length(text, numchars):
    result = pretty_print(text, numchars)
    assert len(result) == numchars, f"pretty_print({text!r}, numchars={numchars}) returned {result!r} with length {len(result)}, expected exactly {numchars}"

if __name__ == "__main__":
    print("Running property-based tests with Hypothesis...")
    print("=" * 60)

    print("\nTesting maybe_truncate()...")
    try:
        test_maybe_truncate_at_most_maxlen()
        print("✓ All tests passed for maybe_truncate()")
    except AssertionError as e:
        print(f"✗ Test failed for maybe_truncate()")
        print(f"  {e}")

    print("\nTesting pretty_print()...")
    try:
        test_pretty_print_produces_exact_length()
        print("✓ All tests passed for pretty_print()")
    except AssertionError as e:
        print(f"✗ Test failed for pretty_print()")
        print(f"  {e}")
```

<details>

<summary>
**Failing input**: `maybe_truncate('00', maxlen=1)` and `pretty_print('00', numchars=1)`
</summary>
```
Running property-based tests with Hypothesis...
============================================================

Testing maybe_truncate()...
✗ Test failed for maybe_truncate()
  maybe_truncate('00', maxlen=1) returned '...' with length 3, expected <= 1

Testing pretty_print()...
✗ Test failed for pretty_print()
  pretty_print('00', numchars=1) returned '...' with length 3, expected exactly 1
```
</details>

## Reproducing the Bug

```python
from xarray.core.formatting import maybe_truncate, pretty_print

# Test maybe_truncate with small maxlen values
print("Testing maybe_truncate():")
print("-" * 40)

# Test case 1: maxlen=1 with string length > 1
result = maybe_truncate('00', maxlen=1)
print(f"maybe_truncate('00', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

# Test case 2: maxlen=2 with longer string
result = maybe_truncate('hello world', maxlen=2)
print(f"maybe_truncate('hello world', maxlen=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 2")
print()

# Test case 3: maxlen=1 with single character (should work)
result = maybe_truncate('a', maxlen=1)
print(f"maybe_truncate('a', maxlen=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 1")
print()

# Test case 4: maxlen=3 with longer string (should work correctly)
result = maybe_truncate('hello world', maxlen=3)
print(f"maybe_truncate('hello world', maxlen=3)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected <= 3")
print()

# Test pretty_print with small numchars values
print("\nTesting pretty_print():")
print("-" * 40)

# Test case 1: numchars=1 with string length > 1
result = pretty_print('00', numchars=1)
print(f"pretty_print('00', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()

# Test case 2: numchars=2 with longer string
result = pretty_print('hello', numchars=2)
print(f"pretty_print('hello', numchars=2)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 2")
print()

# Test case 3: numchars=1 with single character (should work)
result = pretty_print('a', numchars=1)
print(f"pretty_print('a', numchars=1)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 1")
print()

# Test case 4: numchars=10 with shorter string (should pad)
result = pretty_print('hi', numchars=10)
print(f"pretty_print('hi', numchars=10)")
print(f"  Result: {result!r}")
print(f"  Length: {len(result)}, expected exactly 10")
```

<details>

<summary>
Contract violations for multiple test cases
</summary>
```
Testing maybe_truncate():
----------------------------------------
maybe_truncate('00', maxlen=1)
  Result: '...'
  Length: 3, expected <= 1

maybe_truncate('hello world', maxlen=2)
  Result: 'hello worl...'
  Length: 13, expected <= 2

maybe_truncate('a', maxlen=1)
  Result: 'a'
  Length: 1, expected <= 1

maybe_truncate('hello world', maxlen=3)
  Result: '...'
  Length: 3, expected <= 3


Testing pretty_print():
----------------------------------------
pretty_print('00', numchars=1)
  Result: '...'
  Length: 3, expected exactly 1

pretty_print('hello', numchars=2)
  Result: 'hell...'
  Length: 7, expected exactly 2

pretty_print('a', numchars=1)
  Result: 'a'
  Length: 1, expected exactly 1

pretty_print('hi', numchars=10)
  Result: 'hi        '
  Length: 10, expected exactly 10
```
</details>

## Why This Is A Bug

This bug violates the documented and expected behavior of these formatting functions:

1. **`pretty_print()` violates its explicit documentation**: The docstring at line 42-44 states: "Given an object `x`, call `str(x)` and format the returned string so that it is numchars long". The function promises to return a string that "is numchars long" - an exact length guarantee. When `numchars < 3` and the input string is longer than `numchars`, the function returns strings with incorrect length.

2. **`maybe_truncate()` violates reasonable expectations**: While this function lacks documentation, its name and `maxlen` parameter create a clear expectation that the output will never exceed `maxlen` characters. The implementation fails this expectation when `maxlen < 3`.

3. **Root cause analysis**: The bug occurs because when truncating, `maybe_truncate()` uses `s[:(maxlen - 3)] + "..."`. For small `maxlen` values:
   - If `maxlen=1`: `s[:(1-3)]` becomes `s[:-2]`, which when combined with `"..."` produces at least 3 characters
   - If `maxlen=2`: `s[:(2-3)]` becomes `s[:-1]`, which when combined with `"..."` produces at least 3 characters
   - The ellipsis ("...") is always 3 characters long, but the code doesn't handle cases where `maxlen < 3`

4. **Cascading failure**: Since `pretty_print()` relies on `maybe_truncate()` for truncation, it inherits the bug and fails to meet its documented exact-length guarantee.

## Relevant Context

These functions are located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/formatting.py` and appear to be internal utility functions used throughout xarray for formatting display output. While they're in the `core` subpackage (suggesting internal use), they are still accessible and have clear contracts that should be upheld.

The functions are used in various places throughout xarray for formatting array representations, variable summaries, and other display purposes. The bug is unlikely to affect normal usage since truncating to 1-2 characters is rare in practice, but it still represents a contract violation that should be fixed for correctness.

Source code location:
- `maybe_truncate`: lines 50-54 of `xarray/core/formatting.py`
- `pretty_print`: lines 41-47 of `xarray/core/formatting.py`

## Proposed Fix

```diff
--- a/xarray/core/formatting.py
+++ b/xarray/core/formatting.py
@@ -49,7 +49,11 @@ def pretty_print(x, numchars: int):

 def maybe_truncate(obj, maxlen=500):
     s = str(obj)
     if len(s) > maxlen:
-        s = s[: (maxlen - 3)] + "..."
+        if maxlen < 3:
+            # For very small maxlen, just truncate without ellipsis
+            s = s[:maxlen]
+        else:
+            s = s[: (maxlen - 3)] + "..."
     return s
```