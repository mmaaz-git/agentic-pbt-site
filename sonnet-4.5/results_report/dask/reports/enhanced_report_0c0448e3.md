# Bug Report: dask.bag.Bag.join AttributeError in Error Message Handling

**Target**: `dask.bag.Bag.join`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `join` method in dask.bag crashes with an AttributeError when attempting to raise a TypeError for invalid input types due to a typo in the error message construction (`__name` instead of `__name__`).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db


@given(st.integers())
def test_join_error_message_with_invalid_type(invalid_input):
    """
    Property: Error messages should be displayable without crashing.

    This test verifies that when join() is called with an invalid type,
    it should raise a TypeError with a proper error message, not crash
    with an AttributeError.
    """
    bag = db.from_sequence([1, 2, 3])

    try:
        bag.join(invalid_input, lambda x: x)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Joined argument must be" in str(e)
    except AttributeError as e:
        if "'type' object has no attribute '__name'" in str(e) or "has no attribute '_Bag__name'" in str(e):
            raise AssertionError(
                f"Bug found! Error message construction failed with AttributeError: {e}"
            )

# Run the test
test_join_error_message_with_invalid_type()
```

<details>

<summary>
**Failing input**: `0` (or any integer value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 20, in test_join_error_message_with_invalid_type
    bag.join(invalid_input, lambda x: x)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bag/core.py", line 1203, in join
    " delayed object, or Iterable, got %s" % type(other).__name
                                             ^^^^^^^^^^^^^^^^^^
AttributeError: type object 'int' has no attribute '_Bag__name'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 31, in <module>
    test_join_error_message_with_invalid_type()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 9, in test_join_error_message_with_invalid_type
    def test_join_error_message_with_invalid_type(invalid_input):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 26, in test_join_error_message_with_invalid_type
    raise AssertionError(
        f"Bug found! Error message construction failed with AttributeError: {e}"
    )
AssertionError: Bug found! Error message construction failed with AttributeError: type object 'int' has no attribute '_Bag__name'
Falsifying example: test_join_error_message_with_invalid_type(
    invalid_input=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.bag as db

# Create a simple dask bag
bag = db.from_sequence([1, 2, 3])

# Try to join with an invalid type (integer)
try:
    result = bag.join(42, lambda x: x)
    print("No error raised - this should not happen!")
except AttributeError as e:
    print(f"AttributeError caught (unexpected): {e}")
except TypeError as e:
    print(f"TypeError caught (expected): {e}")
```

<details>

<summary>
AttributeError instead of expected TypeError
</summary>
```
AttributeError caught (unexpected): type object 'int' has no attribute '_Bag__name'
```
</details>

## Why This Is A Bug

This violates expected behavior because the `join` method is designed to validate input types and raise a helpful TypeError when invalid arguments are provided. The code at lines 1200-1205 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bag/core.py` specifically checks if the `other` argument is an Iterable and attempts to raise a TypeError with an informative message if it's not.

However, due to a typo on line 1203 (`type(other).__name` instead of `type(other).__name__`), the error message construction itself crashes with an AttributeError. This prevents the intended TypeError from being raised and instead exposes an implementation bug to the user. The error message `"type object 'int' has no attribute '_Bag__name'"` is confusing because:

1. It incorrectly suggests the attribute name is `_Bag__name` (due to Python's name mangling with the single underscore)
2. It doesn't explain what the actual problem is (passing an integer to join instead of an iterable)
3. It raises the wrong exception type (AttributeError instead of TypeError)

The documentation and code intention clearly show that invalid types should result in a TypeError with the message: "Joined argument must be single-partition Bag, delayed object, or Iterable, got [type name]".

## Relevant Context

The `join` method in dask.bag is meant to perform SQL-like join operations between a Bag and another collection (Iterable, single-partition Bag, or Delayed object). The method includes type validation to ensure the `other` parameter is one of these accepted types.

The bug occurs in the validation error path, specifically when a non-iterable, non-Bag, non-Delayed object is passed. This is a common mistake users might make when learning the API or accidentally passing the wrong variable type.

Relevant code location: [dask/bag/core.py:1200-1205](https://github.com/dask/dask/blob/main/dask/bag/core.py#L1200-L1205)

Python's built-in type attributes use double underscores on both sides (e.g., `__name__`, `__class__`, `__dict__`). The single underscore version `__name` does not exist as an attribute of type objects, which is why the AttributeError occurs.

## Proposed Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -1200,7 +1200,7 @@ class Bag(DaskMethodsMixin):
         elif not isinstance(other, Iterable):
             msg = (
                 "Joined argument must be single-partition Bag, "
-                " delayed object, or Iterable, got %s" % type(other).__name
+                " delayed object, or Iterable, got %s" % type(other).__name__
             )
             raise TypeError(msg)
```