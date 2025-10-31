# Bug Report: Cython.Utils.normalise_float_repr - Malformed Output for Negative Floats

**Target**: `Cython.Utils.normalise_float_repr`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalise_float_repr` function produces syntactically invalid float strings when processing negative numbers with negative exponents, embedding the minus sign within the digit string rather than at the beginning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e50, max_value=1e50))
@settings(max_examples=1000)
def test_round_trip_property(f):
    """
    Property: normalise_float_repr should preserve the numerical value.
    For any valid float string, the normalized form should represent the same number.
    """
    float_str = str(f)
    result = normalise_float_repr(float_str)

    try:
        parsed_result = float(result)
    except ValueError as e:
        print(f"\nValueError when parsing result: {e}")
        print(f"Input: {float_str}")
        print(f"Result: {result}")
        raise AssertionError(f"Result '{result}' is not a valid float string")

    assert parsed_result == float(float_str), \
        f"Value changed: {float_str} -> {result} ({float(float_str)} != {parsed_result})"

if __name__ == "__main__":
    test_round_trip_property()
```

<details>

<summary>
**Failing input**: `-2.008974834100108e-30`
</summary>
```
ValueError when parsing result: could not convert string to float: '.0000000000000000000000000000-2008974834100108'
Input: -2.008974834100108e-30
Result: .0000000000000000000000000000-2008974834100108

ValueError when parsing result: could not convert string to float: '.0000000000000000000000000000000000000000000000000000000000000000-8748449926933124'
Input: -8.748449926933124e-66
Result: .0000000000000000000000000000000000000000000000000000000000000000-8748449926933124

ValueError when parsing result: could not convert string to float: '.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-8387314020247791'
Input: -8.387314020247791e-216
Result: .0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000-8387314020247791

ValueError when parsing result: could not convert string to float: '.0000000000000000000000000000-2008974834100108'
Input: -2.008974834100108e-30
Result: .0000000000000000000000000000-2008974834100108

ValueError when parsing result: could not convert string to float: '.0000000000000000000000000000-2008974834100108'
Input: -2.008974834100108e-30
Result: .0000000000000000000000000000-2008974834100108
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 28, in <module>
  |     test_round_trip_property()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 6, in test_round_trip_property
  |     min_value=-1e50, max_value=1e50))
  |     ^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 17, in test_round_trip_property
    |     parsed_result = float(result)
    | ValueError: could not convert string to float: '.0000000000000000000000000000-2008974834100108'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 22, in test_round_trip_property
    |     raise AssertionError(f"Result '{result}' is not a valid float string")
    | AssertionError: Result '.0000000000000000000000000000-2008974834100108' is not a valid float string
    | Falsifying example: test_round_trip_property(
    |     f=-2.008974834100108e-30,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/35/hypo.py:18
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 24, in test_round_trip_property
    |     assert parsed_result == float(float_str), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Value changed: 2.220446049250313e-16 -> 2.000000000000000220446049250313 (2.220446049250313e-16 != 2.0)
    | Falsifying example: test_round_trip_property(
    |     f=2.220446049250313e-16,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/35/hypo.py:25
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import normalise_float_repr

test_cases = [
    "-1e-10",
    "-0.00001",
    "-1.5e-5",
    "-1.1754943508222875e-38",
]

for float_str in test_cases:
    result = normalise_float_repr(float_str)
    print(f"{float_str:30} -> {result}")
    try:
        parsed = float(result)
        print(f"  Parsed successfully as: {parsed}")
    except ValueError as e:
        print(f"  ERROR: '{result}' is not a valid float!")
        print(f"  ValueError: {e}")
```

<details>

<summary>
Output shows syntactically invalid float strings for negative numbers
</summary>
```
-1e-10                         -> .00000000-1
  ERROR: '.00000000-1' is not a valid float!
  ValueError: could not convert string to float: '.00000000-1'
-0.00001                       -> -0.00001
  Parsed successfully as: -1e-05
-1.5e-5                        -> .000-15
  ERROR: '.000-15' is not a valid float!
  ValueError: could not convert string to float: '.000-15'
-1.1754943508222875e-38        -> .000000000000000000000000000000000000-11754943508222875
  ERROR: '.000000000000000000000000000000000000-11754943508222875' is not a valid float!
  ValueError: could not convert string to float: '.000000000000000000000000000000000000-11754943508222875'
```
</details>

## Why This Is A Bug

This violates the function's expected behavior in multiple critical ways:

1. **Syntactic Invalidity**: The function produces malformed float strings like `.00000000-1` that cannot be parsed by Python's `float()` function. The minus sign is embedded within the digit string instead of being placed at the beginning.

2. **Test Suite Expectations**: The existing test suite at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tests/TestCythonUtils.py:198` explicitly verifies that `float(result)` succeeds, confirming that the output must be a valid float literal.

3. **Documented Purpose Violation**: The function's docstring states it generates normalized strings "to allow string comparisons." Syntactically invalid float strings cannot fulfill this purpose.

4. **Compiler Usage Impact**: The function is used in `Cython/Compiler/ExprNodes.py` for compile-time constant evaluation of DEF constants. Invalid output would cause compilation failures when negative float constants are used.

5. **Incomplete Sign Handling**: The bug occurs because the function processes the string with `float_str.lower().lstrip('0')` without first extracting the minus sign. This causes the sign to be treated as part of the digit manipulation, resulting in its incorrect placement in the final output.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py:660-687` and is used by the Cython compiler for normalizing float representations during compile-time constant evaluation.

Key observations:
- All existing test cases in TestCythonUtils.py only test positive numbers (lines 170-192)
- The function correctly handles some negative inputs (e.g., `-0.00001`) when no exponent manipulation is needed
- The bug specifically manifests when negative numbers require decimal point shifting due to negative exponents

The issue also affects positive floats with precision problems (as shown in the Hypothesis test with `2.220446049250313e-16`), though the primary concern is the syntactically invalid output for negative numbers.

## Proposed Fix

```diff
 def normalise_float_repr(float_str):
     """
     Generate a 'normalised', simple digits string representation of a float value
     to allow string comparisons.  Examples: '.123', '123.456', '123.'
     """
-    str_value = float_str.lower().lstrip('0')
+    str_value = float_str.lower()
+
+    # Handle negative sign separately
+    sign = ''
+    if str_value.startswith('-'):
+        sign = '-'
+        str_value = str_value[1:]
+
+    # Strip leading zeros after extracting sign
+    str_value = str_value.lstrip('0')

     exp = 0
     if 'E' in str_value or 'e' in str_value:
         str_value, exp = str_value.split('E' if 'E' in str_value else 'e', 1)
         exp = int(exp)

     if '.' in str_value:
         num_int_digits = str_value.index('.')
         str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
     else:
         num_int_digits = len(str_value)
     exp += num_int_digits

     result = (
         str_value[:exp]
         + '0' * (exp - len(str_value))
         + '.'
         + '0' * -exp
         + str_value[exp:]
     ).rstrip('0')

-    return result if result != '.' else '.0'
+    normalized = result if result != '.' else '.0'
+    return sign + normalized
```