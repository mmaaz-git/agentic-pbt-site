# Bug Report: pandas.core.computation.eval Whitespace-Only Expression Validation

**Target**: `pandas.core.computation.eval.eval`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pd.eval()` function returns `None` for whitespace-only expressions instead of raising `ValueError`, violating the documented contract and creating inconsistent behavior compared to empty string handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, HealthCheck, example
import pandas as pd
import pytest

# Strategy for generating whitespace-only strings
whitespace_strategy = st.text(alphabet=' \t\n\r', min_size=1, max_size=20)

@given(whitespace_strategy)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@example("   ")  # Basic spaces
@example("\t\t")  # Tabs
@example("\n\n")  # Newlines
@example("   \n\t  ")  # Mixed whitespace
def test_eval_whitespace_only_expression_should_raise(whitespace_expr):
    """Test that pd.eval raises ValueError for whitespace-only expressions"""
    # Only test if the string is truly whitespace-only
    if whitespace_expr.strip() == "":
        with pytest.raises(ValueError, match="expr cannot be an empty string"):
            pd.eval(whitespace_expr)

# Run the test
if __name__ == "__main__":
    # Run the test and capture the failure
    test_eval_whitespace_only_expression_should_raise()
```

<details>

<summary>
**Failing input**: `"   "`, `"\t\t"`, `"\n\n"`, `"   \n\t  "`
</summary>
```
Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 24, in <module>
  |     test_eval_whitespace_only_expression_should_raise()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 9, in test_eval_whitespace_only_expression_should_raise
  |     @settings(suppress_health_check=[HealthCheck.filter_too_much])
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | BaseExceptionGroup: Hypothesis found 4 distinct failures in explicit examples. (4 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 18, in test_eval_whitespace_only_expression_should_raise
    |     with pytest.raises(ValueError, match="expr cannot be an empty string"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_eval_whitespace_only_expression_should_raise(
    |     whitespace_expr='   ',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 18, in test_eval_whitespace_only_expression_should_raise
    |     with pytest.raises(ValueError, match="expr cannot be an empty string"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_eval_whitespace_only_expression_should_raise(
    |     whitespace_expr='\t\t',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 18, in test_eval_whitespace_only_expression_should_raise
    |     with pytest.raises(ValueError, match="expr cannot be an empty string"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_eval_whitespace_only_expression_should_raise(
    |     whitespace_expr='\n\n',
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 18, in test_eval_whitespace_only_expression_should_raise
    |     with pytest.raises(ValueError, match="expr cannot be an empty string"):
    |          ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    |     fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    |     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    |     raise Failed(msg=reason, pytrace=pytrace)
    | Failed: DID NOT RAISE <class 'ValueError'>
    | Falsifying explicit example: test_eval_whitespace_only_expression_should_raise(
    |     whitespace_expr='   \n\t  ',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Test whitespace-only expression
result = pd.eval("   \n\t  ")
print(f"Result of pd.eval('   \\n\\t  '): {result}")
print(f"Type of result: {type(result)}")

# Test empty string for comparison
try:
    pd.eval("")
    print("pd.eval('') did not raise an error")
except ValueError as e:
    print(f"pd.eval('') correctly raises ValueError: {e}")

# Test various whitespace combinations
test_cases = [
    '   ',           # spaces only
    '\t\t',          # tabs only
    '\n\n',          # newlines only
    ' \t\n ',        # mixed whitespace
    '    \n\t\n   '  # complex whitespace
]

print("\nTesting various whitespace combinations:")
for test in test_cases:
    result = pd.eval(test)
    print(f"pd.eval({repr(test)}): {result}")
```

<details>

<summary>
Output demonstrating the bug
</summary>
```
Result of pd.eval('   \n\t  '): None
Type of result: <class 'NoneType'>
pd.eval('') correctly raises ValueError: expr cannot be an empty string

Testing various whitespace combinations:
pd.eval('   '): None
pd.eval('\t\t'): None
pd.eval('\n\n'): None
pd.eval(' \t\n '): None
pd.eval('    \n\t\n   '): None
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Inconsistent API behavior**: `pd.eval("")` correctly raises `ValueError: expr cannot be an empty string`, but `pd.eval("   ")` returns `None`. This inconsistency is confusing for users who expect similar treatment of semantically empty expressions.

2. **Violates documented contract**: The internal function `_check_expression()` has a docstring stating it should "Make sure an expression is not an empty string" and should raise `ValueError` "if expr is an empty string". A whitespace-only string becomes empty after stripping, which is what happens during processing.

3. **Silent failure instead of explicit error**: Returning `None` for invalid input violates the principle of fail-fast. Users expect either a valid evaluation result or an exception for invalid input, not a silent `None` return that could mask bugs.

4. **Semantic emptiness ignored**: From a Python expression evaluation perspective, whitespace-only strings contain no evaluable expression. They are semantically equivalent to empty strings and should be treated the same way.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/computation/eval.py`:

- Line 306: `_check_expression(expr)` is called on the original string, which passes for whitespace-only strings since `if not expr:` evaluates to `False` for non-empty strings.
- Line 307: `exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]` creates an empty list for whitespace-only input.
- Lines 326-363: The loop `for expr in exprs:` never executes when `exprs` is empty.
- Line 322: `ret = None` is the default value that gets returned.

The `_check_expression` function at line 108 uses `if not expr:` which only catches truly empty strings, not whitespace-only ones.

## Proposed Fix

```diff
--- a/pandas/core/computation/eval.py
+++ b/pandas/core/computation/eval.py
@@ -305,6 +305,8 @@ def eval(
     if isinstance(expr, str):
         _check_expression(expr)
         exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
+        if not exprs:
+            raise ValueError("expr cannot be an empty string")
     else:
         # ops.BinOp; for internal compat, not intended to be passed by users
         exprs = [expr]
```