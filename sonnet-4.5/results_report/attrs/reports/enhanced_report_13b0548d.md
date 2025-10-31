# Bug Report: attrs cmp_using Grammatical Errors in Error Message

**Target**: `attrs.cmp_using`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cmp_using()` function contains two grammatical errors in its error message when `eq` is not provided but ordering functions are: "define" should be "defined" and "is order" should be "in order".

## Property-Based Test

```python
import attrs
from attrs import cmp_using
import pytest
from hypothesis import given, strategies as st

@given(st.just(None))  # Using a simple strategy since we're testing error messages
def test_cmp_using_error_message_typo(dummy):
    """Test that cmp_using raises ValueError with typo in error message when eq is missing."""
    with pytest.raises(ValueError, match="eq must be define"):
        cmp_using(lt=lambda a, b: a < b)

if __name__ == "__main__":
    # Run the test to show it catches the typo
    from hypothesis import find
    try:
        # Use hypothesis to find an example
        find(st.just(None), lambda x: test_cmp_using_error_message_typo() or True)
    except:
        # Just run the test directly
        try:
            cmp_using(lt=lambda a, b: a < b)
        except ValueError as e:
            print(f"ValueError raised with message: {e}")
            if "eq must be define" in str(e):
                print("Test confirmed: The typo 'eq must be define' exists in the error message")
```

<details>

<summary>
**Failing input**: `None` (dummy value - the test always triggers the error)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/52
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_cmp_using_error_message_typo PASSED                        [100%]
============================ Hypothesis Statistics =============================

hypo.py::test_cmp_using_error_message_typo:

  - during generate phase (0.00 seconds):
    - Typical runtimes: < 1ms, of which < 1ms in data generation
    - 1 passing examples, 0 failing examples, 0 invalid examples

  - Stopped because nothing left to do


============================== 1 passed in 0.01s ===============================
```
</details>

## Reproducing the Bug

```python
from attrs import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(e)
```

<details>

<summary>
ValueError: grammatical errors in error message
</summary>
```
eq must be define is order to complete ordering from lt, le, gt, ge.
```
</details>

## Why This Is A Bug

This violates professional quality standards for user-facing error messages. The error message in `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_cmp.py` at line 105 contains two clear grammatical errors:

1. **"define" should be "defined"**: This is a basic grammatical error where the past participle is needed after "must be"
2. **"is order" should be "in order"**: This is a typo missing the letter "n" in the preposition "in"

According to the documentation, `cmp_using` creates comparison methods and "The resulting class will have a full set of ordering methods if at least one of {lt, le, gt, ge} and eq are provided." When `eq` is missing while providing partial ordering functions, the function correctly raises a `ValueError` as required by Python's `functools.total_ordering` (which needs `__eq__` to be defined). However, the error message text itself contains these unprofessional typos.

## Relevant Context

The error occurs at line 105 in `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_cmp.py`. The surrounding code shows this is an intentional early error to provide a better stack trace when `functools.total_ordering` would fail due to missing `__eq__`:

```python
# Lines 101-107 from attr/_cmp.py
if 0 < num_order_functions < 4:
    if not has_eq_function:
        # functools.total_ordering requires __eq__ to be defined,
        # so raise early error here to keep a nice stack.
        msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
        raise ValueError(msg)
    type_ = functools.total_ordering(type_)
```

The functionality is correct - it properly prevents incomplete ordering definitions. Only the error message text needs correction for professional quality.

Documentation: https://www.attrs.org/en/stable/api.html#attrs.cmp_using

## Proposed Fix

```diff
--- a/attr/_cmp.py
+++ b/attr/_cmp.py
@@ -102,7 +102,7 @@ def cmp_using(
         if not has_eq_function:
             # functools.total_ordering requires __eq__ to be defined,
             # so raise early error here to keep a nice stack.
-            msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
+            msg = "eq must be defined in order to complete ordering from lt, le, gt, ge."
             raise ValueError(msg)
         type_ = functools.total_ordering(type_)
```