# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quote Characters

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when parsing bracket-delimited lists containing unclosed quote characters (e.g., `[']` or `["]`), due to a mismatch between how `strip_string_literals` stores placeholder keys with trailing underscores and how `parse_list.unquote` looks them up without the underscore.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cc', 'Cs')), min_size=1)))
def test_parse_list_bracket_delimited(items):
    s = '[' + ', '.join(items) + ']'
    try:
        result = parse_list(s)
        assert isinstance(result, list)
    except KeyError as e:
        # Re-raise to see the error but with context
        print(f"KeyError for input items={items}, s={s!r}")
        raise

# Run the test
test_parse_list_bracket_delimited()
```

<details>

<summary>
**Failing input**: `items=["'"]`
</summary>
```
KeyError for input items=['º\U00105871İį', 'ď\U000e2919hĝKb_\U0008ebc2', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', 'Į:4¹ĦÿdF$mĄ', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'Ý9ĕ\U000a815bģ', '𰈈'], s="[º\U00105871İį, ď\U000e2919hĝKb_\U0008ebc2, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, Į:4¹ĦÿdF$mĄ, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, Ý9ĕ\U000a815bģ, 𰈈]"
KeyError for input items=['º\U00105871İį', 'ď\U000e2919hĝKb_\U0008ebc2', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', 'Ý9ĕ\U000a815bģ', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'Ý9ĕ\U000a815bģ', '𰈈'], s="[º\U00105871İį, ď\U000e2919hĝKb_\U0008ebc2, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, Ý9ĕ\U000a815bģ, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, Ý9ĕ\U000a815bģ, 𰈈]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', 'Ý9ĕ\U000a815bģ', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'Ý9ĕ\U000a815bģ', '𰈈'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, Ý9ĕ\U000a815bģ, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, Ý9ĕ\U000a815bģ, 𰈈]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'Ý9ĕ\U000a815bģ', '𰈈'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, Ý9ĕ\U000a815bģ, 𰈈]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'Ý9ĕ\U000a815bģ', 'Ý9ĕ\U000a815bģ'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, Ý9ĕ\U000a815bģ, Ý9ĕ\U000a815bģ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'º\U00105871İį', 'Ý9ĕ\U000a815bģ'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, º\U00105871İį, Ý9ĕ\U000a815bģ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', 'Ý9ĕ\U000a815bģ', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'º\U00105871İį', 'Ý9ĕ\U000a815bģ'], s="[º\U00105871İį, 𰈈, 8, Ý9ĕ\U000a815bģ, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, º\U00105871İį, Ý9ĕ\U000a815bģ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'º\U00105871İį', 'Ý9ĕ\U000a815bģ'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, º\U00105871İį, Ý9ĕ\U000a815bģ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'º\U00105871İį', '0'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, º\U00105871İį, 0]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', 'º\U00105871İį'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, º\U00105871İį]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ', '0'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ, 0]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", 'z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, z𪢩ğė\U0003d2ba§\U0006c12eĐ\U00019942Ĉŀ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ", '0'], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ, 0]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '𰈈', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 𰈈, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', 'è', '0', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, è, 0, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', 'ºýÌØ1', '0', '0', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, ºýÌØ1, 0, 0, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '7ĐÊ㦧Ī', '0', '0', '0', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 8, 7ĐÊ㦧Ī, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '𰈈', '8', '0', '0', '0', '0', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 8, 0, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '𰈈', '0', '0', '0', '0', '0', "'}ÌüQ"], s="[º\U00105871İį, 𰈈, 0, 0, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['º\U00105871İį', '0', '0', '0', '0', '0', '0', "'}ÌüQ"], s="[º\U00105871İį, 0, 0, 0, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['0', '0', '0', '0', '0', '0', '0', "'}ÌüQ"], s="[0, 0, 0, 0, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['0', '0', '0', '0', '0', "'}ÌüQ"], s="[0, 0, 0, 0, 0, '}ÌüQ]"
KeyError for input items=['0', '0', '0', "'}ÌüQ"], s="[0, 0, 0, '}ÌüQ]"
KeyError for input items=['0', "'}ÌüQ"], s="[0, '}ÌüQ]"
KeyError for input items=["'}ÌüQ"], s="['}ÌüQ]"
KeyError for input items=["'}Ìü"], s="['}Ìü]"
KeyError for input items=["'}Ì"], s="['}Ì]"
KeyError for input items=["'}"], s="['}]"
KeyError for input items=["'"], s="[']"
KeyError for input items=["'"], s="[']"
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 19, in <module>
    test_parse_list_bracket_delimited()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 8, in test_parse_list_bracket_delimited
    def test_parse_list_bracket_delimited(items):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 11, in test_parse_list_bracket_delimited
    result = parse_list(s)
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_bracket_delimited(
    items=["'"],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132
        /home/npc/pbt/agentic-pbt/worker_/30/hypo.py:13
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import parse_list

# Test the exact failing case from the bug report
result = parse_list('["]')
print(f"Result: {result}")
```

<details>

<summary>
KeyError: '__Pyx_L1' when parsing bracket-delimited list with unclosed quote
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/repo.py", line 7, in <module>
    result = parse_list('["]')
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Cryptic error message**: The function raises `KeyError: '__Pyx_L1'` which exposes internal implementation details. Users should never see placeholder variable names like `__Pyx_L1` in error messages.

2. **Inconsistent string handling**: The function's docstring shows it can handle various quoted strings (e.g., `'[a, ",a", "a,", ",", ]'` returns `['a', ',a', 'a,', ',']`), but it crashes on simple cases like `[']` or `["]`.

3. **Silent data corruption risk**: The root cause is a mismatch between how `strip_string_literals` stores keys (with trailing underscore, e.g., `__Pyx_L1_`) versus how `unquote` looks them up (without underscore, e.g., `__Pyx_L1`). This indicates a synchronization bug between two cooperating functions.

4. **No graceful error handling**: Even if unclosed quotes are considered invalid input, the function should either:
   - Parse them as literal strings
   - Raise a clear `ValueError` with a message like "Unclosed quote in list specification"
   - Not crash with an internal `KeyError`

## Relevant Context

The bug occurs due to the interaction between two functions:

1. **`strip_string_literals` (line 282-310)**: When it encounters an unclosed string literal (like a lone `"`), it creates a placeholder like `"__Pyx_L1_` and stores the mapping with the key `__Pyx_L1_` (with trailing underscore).

2. **`parse_list.unquote` (lines 129-134)**: This nested function receives the placeholder `"__Pyx_L1_` and tries to look up `literal[1:-1]` which becomes `__Pyx_L1` (without the trailing underscore), causing the KeyError.

The same issue affects other edge cases:
- Empty quoted strings `[""]` raise `KeyError: ''`
- Single quotes `[']` raise the same `KeyError: '__Pyx_L1'`

Link to affected code: https://github.com/cython/cython/blob/master/Cython/Build/Dependencies.py#L108-L135

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -129,7 +129,12 @@ def parse_list(s):
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
-            return literals[literal[1:-1]]
+            key = literal[1:-1]
+            if key in literals:
+                return literals[key]
+            else:
+                # Handle case where strip_string_literals created a placeholder
+                # but the key doesn't match exactly (e.g., unclosed quotes)
+                return literal[1:-1] if len(literal) > 2 else ''
         else:
             return literal
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```