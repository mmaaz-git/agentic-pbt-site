# Bug Report: attrs cmp_using Error Message Contains Grammatical Errors

**Target**: `attr._cmp.cmp_using`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cmp_using()` function contains grammatical errors in its error message when `eq` is not provided alongside ordering functions (`lt`, `le`, `gt`, `ge`).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from attr._cmp import cmp_using

@given(st.sampled_from([lambda a, b: a < b, lambda a, b: a > b]))
def test_cmp_using_error_message_grammar(lt_func):
    try:
        cmp_using(lt=lt_func)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "must be defined in order" in error_msg, \
            f"Error message has typos: {error_msg}"

if __name__ == "__main__":
    test_cmp_using_error_message_grammar()
```

<details>

<summary>
**Failing input**: `lambda a, b: a < b`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 10, in test_cmp_using_error_message_grammar
    cmp_using(lt=lt_func)
    ~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_cmp.py", line 106, in cmp_using
    raise ValueError(msg)
ValueError: eq must be define is order to complete ordering from lt, le, gt, ge.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 18, in <module>
    test_cmp_using_error_message_grammar()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 8, in test_cmp_using_error_message_grammar
    def test_cmp_using_error_message_grammar(lt_func):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 14, in test_cmp_using_error_message_grammar
    assert "must be defined in order" in error_msg, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Error message has typos: eq must be define is order to complete ordering from lt, le, gt, ge.
Falsifying example: test_cmp_using_error_message_grammar(
    lt_func=lambda a, b: a < b,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from attr._cmp import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(f"Error message: {e}")
```

<details>

<summary>
ValueError raised with grammatically incorrect error message
</summary>
```
Error message: eq must be define is order to complete ordering from lt, le, gt, ge.
```
</details>

## Why This Is A Bug

The error message at line 105 of `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_cmp.py` contains two grammatical errors that violate professional standards for error messages:

1. **"define" should be "defined"** - The past participle form is required after "must be"
2. **"is order" should be "in order"** - The preposition "in" is missing

The current message reads: `"eq must be define is order to complete ordering from lt, le, gt, ge."`

The correct message should read: `"eq must be defined in order to complete ordering from lt, le, gt, ge."`

This violates the contract of professional software to provide clear, grammatically correct error messages to users. While the intent is still understandable, the typos could confuse non-native English speakers and reflects poorly on the library's quality.

## Relevant Context

The `cmp_using()` function is designed to create comparison classes that can be used with attrs fields. According to the documentation at line 26-27, "The resulting class will have a full set of ordering methods if at least one of `{lt, le, gt, ge}` and `eq` are provided."

When a user provides partial ordering functions (like `lt`, `le`, `gt`, or `ge`) without providing `eq`, the function needs to raise an error because Python's `functools.total_ordering` decorator (which is used at line 107) requires `__eq__` to be defined. The comment at lines 103-104 explicitly states: "functools.total_ordering requires __eq__ to be defined, so raise early error here to keep a nice stack."

The error is raised correctly and at the right time - the only issue is the grammatical mistakes in the error message text itself. This is a quality-of-life issue that impacts the professional presentation of error messages to developers using the library.

Documentation link: The attrs.cmp_using() function is part of the attrs comparison customization API.
Code location: `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_cmp.py:105`

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