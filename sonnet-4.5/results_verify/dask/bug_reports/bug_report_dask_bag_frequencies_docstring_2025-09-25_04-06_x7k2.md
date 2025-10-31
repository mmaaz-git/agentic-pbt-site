# Bug Report: dask.bag.Bag.frequencies Invalid Python Syntax in Docstring

**Target**: `dask.bag.Bag.frequencies`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `frequencies` method in dask.bag has a syntax error in its docstring example. The expected output shows `{'Alice': 2, 'Bob', 1}` which is invalid Python syntax (missing colon after `'Bob'`). This should be `{'Alice': 2, 'Bob': 1}`.

## Property-Based Test

While this is a documentation bug and doesn't affect runtime behavior, we can verify the correct behavior:

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20))
def test_frequencies_returns_valid_dict(items):
    """
    Property: frequencies() should return valid dict-like key-value pairs.

    This verifies that the actual behavior matches what the docstring
    *intends* to show (valid dict syntax), not what it actually shows
    (invalid syntax).
    """
    bag = db.from_sequence(items)
    result = dict(bag.frequencies())

    for key, value in result.items():
        assert isinstance(key, str), f"Key should be string, got {type(key)}"
        assert isinstance(value, int), f"Value should be int, got {type(value)}"
        assert value > 0, f"Frequency should be positive, got {value}"
```

## Reproducing the Bug

The bug is in the docstring itself. Looking at line 938 of `dask/bag/core.py`:

```python
def frequencies(self, split_every=None, sort=False):
    """Count number of occurrences of each distinct element.

    >>> import dask.bag as db
    >>> b = db.from_sequence(['Alice', 'Bob', 'Alice'])
    >>> dict(b.frequencies())       # doctest: +SKIP
    {'Alice': 2, 'Bob', 1}          # <-- SYNTAX ERROR HERE
    """
```

To demonstrate that this is invalid Python syntax:

```python
>>> {'Alice': 2, 'Bob', 1}
  File "<stdin>", line 1
    {'Alice': 2, 'Bob', 1}
                     ^
SyntaxError: invalid syntax
```

The correct syntax should be:

```python
>>> {'Alice': 2, 'Bob': 1}
{'Alice': 2, 'Bob': 1}
```

## Why This Is A Bug

This is a **documentation bug** that violates the API contract:

1. **Misleading documentation**: Users reading the docstring see invalid Python syntax as the expected output
2. **Copy-paste errors**: Users who copy the example from the docstring will get a syntax error if they try to create that dict
3. **Confusing for beginners**: New users might think this is some special dask syntax or get confused about Python dict syntax
4. **Documentation quality**: Professional libraries should have accurate, syntactically correct examples
5. **Marked as `# doctest: +SKIP`**: This suggests the example was never actually tested, which is why the bug wasn't caught

While this doesn't affect the runtime behavior of the `frequencies()` method (which works correctly), it degrades the quality of the API documentation and can confuse users.

## Impact Assessment

**Severity: Low**
- **User Impact**: Confuses users reading documentation, but doesn't affect functionality
- **Frequency**: Affects anyone reading the docstring for this method
- **Debuggability**: May confuse beginners about Python dict syntax
- **Documentation**: Undermines trust in documentation accuracy

## Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -935,7 +935,7 @@ class Bag(DaskMethodsMixin):
         >>> import dask.bag as db
         >>> b = db.from_sequence(['Alice', 'Bob', 'Alice'])
         >>> dict(b.frequencies())       # doctest: +SKIP
-        {'Alice': 2, 'Bob', 1}
+        {'Alice': 2, 'Bob': 1}
         """
         result = self.reduction(
             frequencies,
```

**Note**: Additionally, the `# doctest: +SKIP` should ideally be removed and the actual output verified, as the correct output would be (in no particular order):

```python
{'Alice': 2, 'Bob': 1}
```

or

```python
{'Bob': 1, 'Alice': 2}
```

Since dict ordering is preserved in Python 3.7+, the exact output depends on insertion order during the frequency counting process.