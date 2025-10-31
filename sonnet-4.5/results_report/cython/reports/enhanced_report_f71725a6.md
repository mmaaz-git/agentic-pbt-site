# Bug Report: Cython.Tempita fill_command py: Prefix Incorrectly Retained Instead of Stripped

**Target**: `Cython.Tempita._tempita.fill_command`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `fill_command` function incorrectly retains the "py:" prefix when parsing command-line arguments, causing all Python-evaluated variables to overwrite each other under the same dictionary key "py:" instead of using their actual variable names.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import string

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=10).filter(str.isidentifier))
@settings(max_examples=100)
def test_fill_command_py_prefix_strips_prefix(var_name):
    """Test that py: prefix is correctly stripped from variable names in fill_command"""
    arg_string = f"py:{var_name}"

    # Simulate the buggy parsing logic from Cython.Tempita.fill_command
    name = arg_string
    if name.startswith('py:'):
        parsed_name = name[:3]  # BUG: This keeps 'py:' instead of removing it

    expected_name = var_name
    actual_name = parsed_name

    assert actual_name == expected_name, f"Variable name should be {expected_name!r}, got {actual_name!r}"

if __name__ == "__main__":
    # Run the test to find a failing example
    print("Running Hypothesis property-based test to find failing inputs...")
    print("=" * 60)
    try:
        test_fill_command_py_prefix_strips_prefix()
        print("Test passed (no bug found)")
    except AssertionError as e:
        print(f"Test failed as expected, demonstrating the bug!")
        print(f"Error: {e}")
```

<details>

<summary>
**Failing input**: `A` (and any other valid Python identifier)
</summary>
```
Running Hypothesis property-based test to find failing inputs...
============================================================
Test failed as expected, demonstrating the bug!
Error: Variable name should be 'A', got 'py:'
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Simulating the exact bug from the fill_command function
# This is the problematic code from line 1072-1075 of Cython/Tempita/_tempita.py

def test_py_prefix_bug():
    """Demonstrate the bug in Cython.Tempita.fill_command py: prefix handling"""

    # Test case 1: Single py: prefixed variable
    print("Test Case 1: Single py: prefixed variable")
    print("-" * 40)

    vars = {}
    arg_string = "py:my_var=42"

    # Simulate parsing the argument (lines 1071-1075)
    name, value = arg_string.split('=', 1)
    print(f"Original argument: {arg_string}")
    print(f"After split: name='{name}', value='{value}'")

    if name.startswith('py:'):
        name = name[:3]  # BUG: This keeps 'py:' instead of removing it
        value = eval(value)

    vars[name] = value

    print(f"Variable name stored in dict: '{name}'")
    print(f"Variable value stored: {value}")
    print(f"Result: vars = {vars}")
    print(f"Expected: vars = {{'my_var': 42}}")
    print(f"Actual:   vars = {vars}")
    print()

    # Test case 2: Multiple py: prefixed variables showing overwrite
    print("Test Case 2: Multiple py: prefixed variables")
    print("-" * 40)

    vars = {}
    args = ["py:x=10", "py:y=20", "py:z=30"]

    print(f"Arguments to parse: {args}")

    for arg_string in args:
        name, value = arg_string.split('=', 1)
        if name.startswith('py:'):
            name = name[:3]  # BUG: All become 'py:'
            value = eval(value)
        vars[name] = value
        print(f"  After parsing '{arg_string}': vars = {vars}")

    print()
    print(f"Expected: vars = {{'x': 10, 'y': 20, 'z': 30}}")
    print(f"Actual:   vars = {vars}")
    print(f"Bug: All variables overwrite each other under key 'py:'!")
    print()

    # Test case 3: Compare with correct implementation
    print("Test Case 3: Correct implementation (what it should be)")
    print("-" * 40)

    vars = {}
    args = ["py:x=10", "py:y=20", "py:z=30"]

    print(f"Arguments to parse: {args}")

    for arg_string in args:
        name, value = arg_string.split('=', 1)
        if name.startswith('py:'):
            name = name[3:]  # FIX: Remove 'py:' prefix correctly
            value = eval(value)
        vars[name] = value
        print(f"  After parsing '{arg_string}': vars = {vars}")

    print()
    print(f"Result with fix: vars = {vars}")
    print(f"This is what the code should produce!")

if __name__ == "__main__":
    test_py_prefix_bug()
```

<details>

<summary>
Demonstrates that all py: prefixed variables incorrectly store under the key 'py:' and overwrite each other
</summary>
```
Test Case 1: Single py: prefixed variable
----------------------------------------
Original argument: py:my_var=42
After split: name='py:my_var', value='42'
Variable name stored in dict: 'py:'
Variable value stored: 42
Result: vars = {'py:': 42}
Expected: vars = {'my_var': 42}
Actual:   vars = {'py:': 42}

Test Case 2: Multiple py: prefixed variables
----------------------------------------
Arguments to parse: ['py:x=10', 'py:y=20', 'py:z=30']
  After parsing 'py:x=10': vars = {'py:': 10}
  After parsing 'py:y=20': vars = {'py:': 20}
  After parsing 'py:z=30': vars = {'py:': 30}

Expected: vars = {'x': 10, 'y': 20, 'z': 30}
Actual:   vars = {'py:': 30}
Bug: All variables overwrite each other under key 'py:'!

Test Case 3: Correct implementation (what it should be)
----------------------------------------
Arguments to parse: ['py:x=10', 'py:y=20', 'py:z=30']
  After parsing 'py:x=10': vars = {'x': 10}
  After parsing 'py:y=20': vars = {'x': 10, 'y': 20}
  After parsing 'py:z=30': vars = {'x': 10, 'y': 20, 'z': 30}

Result with fix: vars = {'x': 10, 'y': 20, 'z': 30}
This is what the code should produce!
```
</details>

## Why This Is A Bug

This bug violates the documented behavior and breaks the intended functionality of the `py:` prefix feature. The issue stems from a clear typo in line 1073 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py`:

1. **Documentation states the intent**: The docstring at lines 1032-1033 explicitly states "Use py:arg=value to set a Python value", indicating that `py:x=42` should create a variable named `x`, not `py:`.

2. **Array slicing error**: The code uses `name[:3]` which extracts the first 3 characters ("py:") instead of `name[3:]` which would remove the prefix and keep the variable name.

3. **Inconsistent with template parsing**: The same module correctly handles the `py:` prefix in templates at line 753 using `expr[3:]`, showing the intended behavior.

4. **Causes data loss**: When multiple `py:` prefixed arguments are provided (e.g., `py:x=10 py:y=20 py:z=30`), they all overwrite each other under the single key "py:", resulting in only the last value being retained.

5. **Breaks template substitution**: Templates expecting variables like `{{x}}` fail to receive their values even when users correctly provide `py:x=value` on the command line.

## Relevant Context

The `fill_command` function is part of Cython's Tempita templating system, used for command-line template processing. The bug specifically affects the command-line interface when users try to pass Python-evaluated values using the documented `py:` prefix syntax.

Key code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:1073`
- Correct implementation for comparison: Same file, line 753 (template parsing)
- Documentation: Same file, lines 1032-1033

The bug has likely existed since the feature was introduced, as it appears to be a simple typo that would immediately break the feature but might go unnoticed if users primarily use string arguments or work directly with templates rather than the command-line interface.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -1070,7 +1070,7 @@ def fill_command(args=None):
             sys.exit(2)
         name, value = value.split('=', 1)
         if name.startswith('py:'):
-            name = name[:3]
+            name = name[3:]
             value = eval(value)
         vars[name] = value
     if template_name == '-':
```