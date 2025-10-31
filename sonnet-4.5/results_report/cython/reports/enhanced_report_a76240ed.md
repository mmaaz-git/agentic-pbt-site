# Bug Report: Cython.Build.Dependencies.parse_list Hash Character Data Corruption

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_list()` function silently corrupts any value containing a hash (`#`) character by treating it as a comment delimiter, replacing everything after `#` with `__Pyx_L1_`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test that discovered the Cython parse_list bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, example
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1)))
@example(['#'])  # Minimal failing example
def test_parse_list_bracket_delimited(items):
    assume(all(item.strip() for item in items))
    assume(all(',' not in item and '"' not in item and "'" not in item for item in items))
    s = '[' + ', '.join(items) + ']'
    result = parse_list(s)
    assert result == items, f"Expected {items}, got {result}"

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test for parse_list...")
    print("This test checks that parse_list correctly handles bracket-delimited lists.")
    print()
    try:
        test_parse_list_bracket_delimited()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed! {e}")
        print("\nThis test fails when items contain '#' characters.")
        print("The minimal failing example found by Hypothesis is: items=['#']")
```

<details>

<summary>
**Failing input**: `items=['#']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/20
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_parse_list_bracket_delimited FAILED                        [100%]

=================================== FAILURES ===================================
______________________ test_parse_list_bracket_delimited _______________________
hypo.py:11: in test_parse_list_bracket_delimited
    @example(['#'])  # Minimal failing example
                   ^^^
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
hypo.py:17: in test_parse_list_bracket_delimited
    assert result == items, f"Expected {items}, got {result}"
E   AssertionError: Expected ['#'], got ['#__Pyx_L1_']
E   assert ['#__Pyx_L1_'] == ['#']
E
E     At index 0 diff: '#__Pyx_L1_' != '#'
E
E     Full diff:
E       [
E     -     '#',
E     +     '#__Pyx_L1_',
E       ]
E   Falsifying explicit example: test_parse_list_bracket_delimited(
E       items=['#'],
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_parse_list_bracket_delimited - AssertionError: Expected ...
============================== 1 failed in 0.17s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython parse_list hash character corruption bug"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import parse_list

print("Testing parse_list with hash (#) characters:")
print("=" * 50)

# Test 1: Single hash character
test1_input = '[#]'
test1_result = parse_list(test1_input)
print(f"Test 1: parse_list({test1_input!r})")
print(f"  Expected: ['#']")
print(f"  Actual:   {test1_result}")
print()

# Test 2: Hash in value
test2_input = '[foo#bar]'
test2_result = parse_list(test2_input)
print(f"Test 2: parse_list({test2_input!r})")
print(f"  Expected: ['foo#bar']")
print(f"  Actual:   {test2_result}")
print()

# Test 3: Multiple items with hash
test3_input = '[libA, libB#version]'
test3_result = parse_list(test3_input)
print(f"Test 3: parse_list({test3_input!r})")
print(f"  Expected: ['libA', 'libB#version']")
print(f"  Actual:   {test3_result}")
print()

# Test 4: Space-separated with hash
test4_input = 'foo#bar baz'
test4_result = parse_list(test4_input)
print(f"Test 4: parse_list({test4_input!r})")
print(f"  Expected: ['foo#bar', 'baz']")
print(f"  Actual:   {test4_result}")
print()

# Test 5: Quoted value with hash (should work correctly)
test5_input = '["lib#version"]'
test5_result = parse_list(test5_input)
print(f"Test 5: parse_list({test5_input!r}) - quoted value")
print(f"  Expected: ['lib#version']")
print(f"  Actual:   {test5_result}")
print()

print("=" * 50)
print("DEMONSTRATION OF BUG:")
print("The hash character (#) is incorrectly treated as a comment delimiter,")
print("corrupting values that contain it. The string after '#' is replaced")
print("with '__Pyx_L1_' which is a placeholder for string literals.")

# Show the assertion failure
try:
    assert parse_list('[foo#bar]') == ['foo#bar'], f"Expected ['foo#bar'], got {parse_list('[foo#bar]')}"
except AssertionError as e:
    print(f"\nAssertion Error: {e}")
```

<details>

<summary>
Output shows data corruption for all test cases with `#` characters
</summary>
```
Testing parse_list with hash (#) characters:
==================================================
Test 1: parse_list('[#]')
  Expected: ['#']
  Actual:   ['#__Pyx_L1_']

Test 2: parse_list('[foo#bar]')
  Expected: ['foo#bar']
  Actual:   ['foo#__Pyx_L1_']

Test 3: parse_list('[libA, libB#version]')
  Expected: ['libA', 'libB#version']
  Actual:   ['libA', 'libB#__Pyx_L1_']

Test 4: parse_list('foo#bar baz')
  Expected: ['foo#bar', 'baz']
  Actual:   ['foo#__Pyx_L1_']

Test 5: parse_list('["lib#version"]') - quoted value
  Expected: ['lib#version']
  Actual:   ['lib#version']

==================================================
DEMONSTRATION OF BUG:
The hash character (#) is incorrectly treated as a comment delimiter,
corrupting values that contain it. The string after '#' is replaced
with '__Pyx_L1_' which is a placeholder for string literals.

Assertion Error: Expected ['foo#bar'], got ['foo#__Pyx_L1_']
```
</details>

## Why This Is A Bug

The `parse_list()` function is used to parse compiler directive values from Cython source files (lines 197-198 in Dependencies.py). These directives specify build configuration like libraries, include directories, and compiler flags:

```python
# distutils: libraries = lib#version
```

The bug occurs because `parse_list()` calls `strip_string_literals()` at line 128, which is designed to parse Python source code and treats `#` as starting a comment. However, the input to `parse_list()` is not Python source code - it's already-extracted directive values that should be treated as data.

This causes silent corruption of legitimate values containing `#` characters, such as:
- Library version tags (e.g., `libfoo#1.2.3`)
- Build variants (e.g., `debug#2`)
- Paths containing `#` characters
- Any identifier using `#` as part of its naming scheme

The corruption is particularly problematic because:
1. It happens silently without warnings or errors
2. The corrupted value `__Pyx_L1_` gives no indication of what went wrong
3. Users won't understand why their builds are failing
4. The behavior is undocumented

## Relevant Context

The `strip_string_literals()` function (lines 282-389) is designed to handle Python source code, where `#` legitimately starts comments. When it encounters `#`, it replaces everything from `#` to the end of the line with a placeholder label like `__Pyx_L1_`.

The function works correctly for its intended purpose (parsing Python source), but using it on already-extracted directive values is incorrect. The directive values have already been extracted from comment lines and should not be treated as Python code.

Key code locations:
- `parse_list()` function: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:108-135`
- Problematic call to `strip_string_literals()`: line 128
- Usage in `DistutilsInfo` class: lines 197-198

Workaround: Values containing `#` can be quoted to avoid corruption (e.g., `["lib#version"]`), but this requires users to know about the bug.

## Proposed Fix

The issue can be fixed by not treating `#` as a comment delimiter when parsing directive values. Here's a minimal fix that handles quoted strings without full Python comment parsing:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -125,7 +125,20 @@ def parse_list(s):
         s = s[1:-1]
         delimiter = ','
     else:
         delimiter = ' '
-    s, literals = strip_string_literals(s)
+
+    # Don't use strip_string_literals which treats # as comments
+    # Instead, just handle quoted strings
+    import re
+    literals = {}
+    counter = 0
+
+    def replace_quotes(m):
+        nonlocal counter
+        counter += 1
+        label = f"__Pyx_L{counter}_"
+        literals[label] = m.group(1) or m.group(2)
+        return label
+
+    s = re.sub(r'"([^"]*)"|\'([^\']*)\'', replace_quotes, s)
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
```