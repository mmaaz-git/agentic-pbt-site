# Bug Report: dask.utils.ndeepmap Silent Data Loss

**Target**: `dask.utils.ndeepmap`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ndeepmap(n, func, seq)` silently discards all elements except the first when `n <= 0` and `seq` is a list with multiple elements. This violates the principle of least surprise and causes silent data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import ndeepmap


def identity(x):
    return x


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_ndeepmap_n0_loses_data(lst):
    """
    Property: ndeepmap should not silently discard data.

    When n=0, the current implementation calls func(seq[0]),
    which loses all elements after the first.
    """
    result = ndeepmap(0, identity, lst)

    # The bug: result only depends on first element
    assert result == lst[0]

    # Demonstrate data loss: changing other elements doesn't affect result
    modified_lst = [lst[0]] + [999] * (len(lst) - 1)
    modified_result = ndeepmap(0, identity, modified_lst)

    # This assertion passes, proving elements lst[1:] are ignored
    assert result == modified_result
```

**Failing input**: `[1, 2, 3]` (any list with 2+ elements)

## Reproducing the Bug

```python
from dask.utils import ndeepmap

def identity(x):
    return x

# Bug demonstration
input_list = [10, 20, 30]
result = ndeepmap(0, identity, input_list)

print(f"Input:  {input_list}")
print(f"Result: {result}")
# Output:
#   Input:  [10, 20, 30]
#   Result: 10

# Elements [20, 30] are silently lost!

# Verify that result doesn't depend on discarded elements
input_modified = [10, 999, 999]
result_modified = ndeepmap(0, identity, input_modified)
assert result == result_modified  # Both return 10
```

## Why This Is A Bug

The current implementation of `ndeepmap` when `n <= 0`:

```python
def ndeepmap(n, func, seq):
    if n == 1:
        return [func(item) for item in seq]
    elif n > 1:
        return [ndeepmap(n - 1, func, item) for item in seq]
    elif isinstance(seq, list):
        return func(seq[0])  # ← BUG: Only processes first element!
    else:
        return func(seq)
```

When `seq` is a list with multiple elements and `n <= 0`, the function returns `func(seq[0])`, silently discarding `seq[1:]`. This is problematic because:

1. **Silent data loss**: Users are not warned that their data is being discarded
2. **Unexpected behavior**: No reasonable user would expect `ndeepmap(0, f, [1,2,3])` to ignore `[2,3]`
3. **Violates function contract**: The docstring doesn't mention this behavior

The existing test shows this is "by design":
```python
L = [1]
assert ndeepmap(0, inc, L) == 2  # inc(L[0]) = inc(1) = 2
```

However, the test only uses single-element lists, hiding the data loss issue.

## Fix

The function should either raise an error or handle all elements. Here's a proposed fix:

```diff
def ndeepmap(n, func, seq):
    """Call a function on every element within a nested container

    >>> def inc(x):
    ...     return x + 1
    >>> L = [[1, 2], [3, 4, 5]]
    >>> ndeepmap(2, inc, L)
    [[2, 3], [4, 5, 6]]
    """
    if n == 1:
        return [func(item) for item in seq]
    elif n > 1:
        return [ndeepmap(n - 1, func, item) for item in seq]
-   elif isinstance(seq, list):
-       return func(seq[0])
+   elif isinstance(seq, list):
+       if len(seq) != 1:
+           raise ValueError(
+               f"ndeepmap called with n={n} on list of length {len(seq)}. "
+               f"Expected a scalar or single-element list."
+           )
+       return func(seq[0])
    else:
        return func(seq)
```

Alternative fix (no error, just apply to whole list):
```diff
-   elif isinstance(seq, list):
-       return func(seq[0])
+   elif isinstance(seq, list):
+       # When n <= 0, treat the entire list as a single object
+       return func(seq)
    else:
        return func(seq)
```