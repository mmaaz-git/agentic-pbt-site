# Bug Report: pandas.util._decorators.deprecate Operator Precedence Validation Bug

**Target**: `pandas.util._decorators.deprecate`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deprecate` function contains an operator precedence bug in its docstring validation logic that causes it to incorrectly accept malformed docstrings lacking a blank line after the summary, violating pandas/NumPy documentation standards.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.util._decorators import deprecate
import pytest


@given(
    st.text(min_size=1),
    st.text(min_size=1),
    st.text(min_size=1),
)
@settings(max_examples=10)
def test_deprecate_rejects_malformed_docstrings(summary, non_empty_after, rest):
    """Test that deprecate rejects docstrings without a blank line after the summary."""
    def bad_alternative():
        pass

    # Create a malformed docstring with no blank line after the summary
    bad_alternative.__doc__ = f"\n{summary}\n{non_empty_after}\n{rest}"

    # This should raise an AssertionError because there's no blank line after the summary
    with pytest.raises(AssertionError):
        deprecate("old", bad_alternative, "1.0")


if __name__ == "__main__":
    print("Running Hypothesis test for deprecate function...")
    print("Testing that deprecate rejects docstrings without a blank line after the summary")
    print("-" * 70)

    try:
        test_deprecate_rejects_malformed_docstrings()
        print("\nAll tests passed! The deprecate function correctly rejects malformed docstrings.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        print("\nThe deprecate function is NOT properly rejecting malformed docstrings.")
        print("This confirms the bug: docstrings without a blank line after the summary")
        print("are being incorrectly accepted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `summary='0', non_empty_after='0', rest='0'`
</summary>
```
Running Hypothesis test for deprecate function...
Testing that deprecate rejects docstrings without a blank line after the summary
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 31, in <module>
    test_deprecate_rejects_malformed_docstrings()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 7, in test_deprecate_rejects_malformed_docstrings
    st.text(min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 21, in test_deprecate_rejects_malformed_docstrings
    with pytest.raises(AssertionError):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'AssertionError'>
Falsifying example: test_deprecate_rejects_malformed_docstrings(
    # The test always failed when commented parts were varied together.
    summary='0',  # or any other generated value
    non_empty_after='0',  # or any other generated value
    rest='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.util._decorators import deprecate


def bad_alternative():
    """
    Summary line
    Next line immediately (no blank line after summary)
    More content here
    """
    pass


def good_alternative():
    """
    Summary line

    Proper blank line after summary
    """
    pass


print("Testing malformed docstring (no blank line after summary)...")
try:
    result = deprecate("old_func", bad_alternative, "1.0.0")
    print("BUG: Malformed docstring was accepted!")
    print(f"Result type: {type(result)}")
except AssertionError as e:
    print("Correctly rejected with error:")
    print(str(e))

print("\nTesting properly formatted docstring...")
try:
    result = deprecate("old_func", good_alternative, "1.0.0")
    print("SUCCESS: Properly formatted docstring was accepted")
    print(f"Result type: {type(result)}")
except AssertionError as e:
    print("ERROR: Good docstring was rejected:")
    print(str(e))
```

<details>

<summary>
BUG: Malformed docstring was incorrectly accepted
</summary>
```
Testing malformed docstring (no blank line after summary)...
BUG: Malformed docstring was accepted!
Result type: <class 'function'>

Testing properly formatted docstring...
SUCCESS: Properly formatted docstring was accepted
Result type: <class 'function'>
```
</details>

## Why This Is A Bug

This violates the expected behavior because the `deprecate` function explicitly states in its error message (lines 71-74 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/_decorators.py`) that it "needs a correctly formatted docstring in the target function (should have a one liner short summary, and opening quotes should be in their own line)".

The code structure at line 83 splits the docstring expecting 4 parts:
```python
empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
```

This implies the expected format is:
- `empty1`: Empty line after opening quotes
- `summary`: The one-line summary
- `empty2`: Empty line after the summary (should be blank)
- `doc_string`: The rest of the documentation

The validation at line 84 attempts to check these conditions:
```python
if empty1 or empty2 and not summary:
    raise AssertionError(doc_error_msg)
```

However, due to Python's operator precedence where `and` binds tighter than `or`, this is incorrectly evaluated as:
```python
if empty1 or (empty2 and not summary):
```

This means when a docstring has:
- `empty1 = ""` (correct - blank line after opening quotes)
- `summary = "Summary line"` (exists)
- `empty2 = "Next line immediately"` (incorrect - should be empty)

The condition evaluates to: `False or (True and False) = False`, so no error is raised despite the malformed format.

This contradicts both:
1. The pandas/NumPy documentation standard which explicitly requires "a blank line after the one-line summary before continuing the docstring"
2. The clear intent of the validation logic to enforce proper formatting

## Relevant Context

The pandas documentation follows NumPy documentation standards which require multi-line docstrings to have:
- Opening quotes on their own line
- A one-line summary
- A blank line after the summary before additional content
- Closing quotes on their own line

The bug allows improperly formatted docstrings to pass validation, which could cause issues with:
- Automatic documentation generation tools that expect NumPy-style formatting
- Consistency across the pandas codebase
- Generated deprecation messages that may not format correctly

The deprecate function is used throughout pandas to mark deprecated functionality, making consistent docstring formatting important for maintaining documentation quality.

## Proposed Fix

```diff
--- a/pandas/util/_decorators.py
+++ b/pandas/util/_decorators.py
@@ -81,7 +81,7 @@ def deprecate(
         if alternative.__doc__.count("\n") < 3:
             raise AssertionError(doc_error_msg)
         empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
-        if empty1 or empty2 and not summary:
+        if empty1 or empty2 or not summary:
             raise AssertionError(doc_error_msg)
         wrapper.__doc__ = dedent(
             f"""
```