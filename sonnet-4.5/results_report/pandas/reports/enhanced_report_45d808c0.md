# Bug Report: Cython.Compiler.PyrexTypes.cap_length Exceeds Maximum Length for Small max_len Values

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length` function fails to respect the `max_len` parameter when `max_len < 17`, producing strings that are longer than the specified maximum length limit.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for the cap_length bug."""

from hypothesis import given, strategies as st, settings
from Cython.Compiler import PyrexTypes


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
       st.integers(min_value=10, max_value=200))
@settings(max_examples=1000)
def test_cap_length_respects_max_len(s, max_len):
    result = PyrexTypes.cap_length(s, max_len)
    assert len(result) <= max_len, f"cap_length({s!r}, {max_len}) returned {result!r} with length {len(result)} > {max_len}"


if __name__ == '__main__':
    test_cap_length_respects_max_len()
```

<details>

<summary>
**Failing input**: `s='00000000000', max_len=10`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 17, in <module>
    test_cap_length_respects_max_len()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 9, in test_cap_length_respects_max_len
    st.integers(min_value=10, max_value=200))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 13, in test_cap_length_respects_max_len
    assert len(result) <= max_len, f"cap_length({s!r}, {max_len}) returned {result!r} with length {len(result)} > {max_len}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: cap_length('00000000000', 10) returned '9c9f57__0000__etc' with length 17 > 10
Falsifying example: test_cap_length_respects_max_len(
    s='00000000000',
    max_len=10,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py:5707
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the cap_length bug."""

from Cython.Compiler import PyrexTypes

# Test case from the bug report
s = '00000000000'
max_len = 10

result = PyrexTypes.cap_length(s, max_len)

print(f"Input string: {s!r}")
print(f"Input length: {len(s)}")
print(f"Max length: {max_len}")
print(f"Result: {result!r}")
print(f"Result length: {len(result)}")
print(f"Expected: <= {max_len}")
print(f"Bug: {len(result)} > {max_len} (should be at most {max_len})")
```

<details>

<summary>
Result exceeds specified max_len
</summary>
```
Input string: '00000000000'
Input length: 11
Max length: 10
Result: '9c9f57__0000__etc'
Result length: 17
Expected: <= 10
Bug: 17 > 10 (should be at most 10)
```
</details>

## Why This Is A Bug

The `cap_length` function is designed to truncate strings that exceed a maximum length while preserving uniqueness through a hash prefix. The function name and its `max_len` parameter establish a clear contract: the returned string should never exceed `max_len` characters.

The bug occurs because the current implementation uses the formula `s[:max_len-17]` to extract a portion of the original string. When `max_len < 17`, this becomes a negative slice index (e.g., `s[-7]` when `max_len=10`), which Python interprets as slicing from the end of the string. This causes the function to include more characters than intended. The final format string `'%s__%s__etc'` then produces:
- 6 characters for the hash prefix
- 2 characters for the first "__"
- Variable number of characters from the incorrectly sliced string
- 5 characters for "__etc"

This results in strings that are always at least 13 characters long (6+2+0+5) even when `max_len` is smaller, violating the function's contract.

## Relevant Context

The function is located in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py` at line 5704.

Currently, all internal uses of this function within Cython use the default `max_len=63`:
- Line 3521: `cap_length("_".join(arg_names))` - for function argument names
- Line 5655: `cap_length('__and_'.join(...))` - for type list identifiers
- Line 5700: `cap_length(re.sub(...))` - for type identifiers

Since all current call sites use the default value of 63, this bug doesn't affect existing Cython functionality. However, the function is part of the public API in the `PyrexTypes` module and could be called by user code or future Cython development with smaller `max_len` values.

## Proposed Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -5704,5 +5704,8 @@ def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
-    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
+    # Format: "hash__truncated__etc" = 6 + 2 + truncated + 5 = 13 + truncated
+    # Ensure we don't exceed max_len
+    available_for_original = max(0, max_len - 13)
+    return '%s__%s__etc' % (hash_prefix, s[:available_for_original])
```