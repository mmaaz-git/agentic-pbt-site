# Bug Report: numpy.f2py.symbolic.eliminate_quotes AssertionError on Unpaired Quotes

**Target**: `numpy.f2py.symbolic.eliminate_quotes`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `eliminate_quotes` function crashes with an AssertionError when given input strings containing unpaired quote characters, which can legitimately occur in Fortran source code (e.g., in comments or malformed strings).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from numpy.f2py import symbolic


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=500)
def test_quote_elimination_round_trip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    reconstructed = symbolic.insert_quotes(new_s, mapping)
    assert s == reconstructed


if __name__ == "__main__":
    # Run the test
    test_quote_elimination_round_trip()
```

<details>

<summary>
**Failing input**: `s="'"`
</summary>
```
Falsifying example: test_quote_elimination_round_trip(
    s="'",
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 15, in <module>
    test_quote_elimination_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_quote_elimination_round_trip
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 8, in test_quote_elimination_round_trip
    new_s, mapping = symbolic.eliminate_quotes(s)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1195, in eliminate_quotes
    assert "'" not in new_s
           ^^^^^^^^^^^^^^^^
AssertionError
```
</details>

## Reproducing the Bug

```python
from numpy.f2py import symbolic

# Test with single double-quote
s = '"'
print(f"Testing with single double-quote: {s!r}")
try:
    new_s, mapping = symbolic.eliminate_quotes(s)
    print(f"Result: new_s={new_s!r}, mapping={mapping}")
except AssertionError as e:
    print(f"AssertionError raised")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test with single single-quote
s = "'"
print(f"Testing with single single-quote: {s!r}")
try:
    new_s, mapping = symbolic.eliminate_quotes(s)
    print(f"Result: new_s={new_s!r}, mapping={mapping}")
except AssertionError as e:
    print(f"AssertionError raised")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
AssertionError on both single and double unpaired quotes
</summary>
```
Testing with single double-quote: '"'
AssertionError raised
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/repo.py", line 7, in <module>
    new_s, mapping = symbolic.eliminate_quotes(s)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1194, in eliminate_quotes
    assert '"' not in new_s
           ^^^^^^^^^^^^^^^^
AssertionError

==================================================

Testing with single single-quote: "'"
AssertionError raised
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/repo.py", line 20, in <module>
    new_s, mapping = symbolic.eliminate_quotes(s)
                     ~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/f2py/symbolic.py", line 1195, in eliminate_quotes
    assert "'" not in new_s
           ^^^^^^^^^^^^^^^^
AssertionError
```
</details>

## Why This Is A Bug

The `eliminate_quotes` function is designed to replace quoted substrings in Fortran/C code with placeholder tokens. However, it uses a regular expression that only matches properly paired quotes (e.g., `"..."` or `'...'`). When the input contains unpaired quotes, the regex doesn't match them, leaving the quotes in the output string. The function then hits assertion statements that check no quotes remain:

```python
assert '"' not in new_s  # Line 1194
assert "'" not in new_s  # Line 1195
```

This violates expected behavior in several ways:

1. **Undocumented precondition**: The function's docstring states it "Replace[s] quoted substrings of input string" without documenting that quotes must be paired.

2. **Inappropriate error type**: AssertionError is an internal implementation detail that shouldn't be exposed to users. Production code should raise proper exceptions like ValueError with descriptive messages.

3. **Real-world impact**: Fortran source code can legitimately contain unpaired quotes, especially in:
   - Comments: `! This is a comment with an unmatched "`
   - Malformed code during development
   - Partial code snippets being analyzed

4. **Inconsistent with insert_quotes**: The inverse function `insert_quotes` doesn't have corresponding assertions, creating an asymmetry in the API.

## Relevant Context

The function is located in `/numpy/f2py/symbolic.py` at lines 1171-1197. The regex pattern used is:

```python
r'({kind}_|)({single_quoted}|{double_quoted})'.format(
    kind=r'\w[\w\d_]*',
    single_quoted=r"('([^'\\]|(\\.))*')",
    double_quoted=r'("([^"\\]|(\\.))*")')
```

This pattern requires quotes to be properly paired and won't match standalone quote characters. The function is part of numpy's f2py module, which is used to wrap Fortran code for use in Python.

Documentation: https://numpy.org/doc/stable/f2py/

## Proposed Fix

Replace the assertions with proper error handling that provides a clear error message:

```diff
--- a/symbolic.py
+++ b/symbolic.py
@@ -1191,8 +1191,11 @@ def eliminate_quotes(s):
         double_quoted=r'("([^"\\]|(\\.))*")'),
         repl, s)

-    assert '"' not in new_s
-    assert "'" not in new_s
+    if '"' in new_s:
+        raise ValueError(f"Unpaired double quote found in input: {s!r}")
+    if "'" in new_s:
+        raise ValueError(f"Unpaired single quote found in input: {s!r}")

     return new_s, d
```

Alternatively, the function could be made more robust by leaving unpaired quotes unchanged or escaping them, depending on the intended use case.