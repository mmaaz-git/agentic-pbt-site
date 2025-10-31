# Bug Report: Cython.Compiler.PyrexTypes.cap_length Violates Length Constraint for Small max_len Values

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cap_length` function fails to enforce its maximum length constraint when `max_len < 17`, returning strings significantly longer than the specified maximum due to incorrect handling of the fixed overhead in the formatted output string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)), st.integers(min_value=10, max_value=200))
def test_cap_length_respects_max(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len

if __name__ == "__main__":
    test_cap_length_respects_max()
```

<details>

<summary>
**Failing input**: `s='00000000000', max_len=10`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 11, in <module>
    test_cap_length_respects_max()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 6, in test_cap_length_respects_max
    def test_cap_length_respects_max(s, max_len):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 8, in test_cap_length_respects_max
    assert len(result) <= max_len
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_cap_length_respects_max(
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
from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00000000000', max_len=10)
print(f"Result: {result!r}")
print(f"Length: {len(result)}")
print(f"Expected max: 10")
print(f"Violates constraint: {len(result) > 10}")
```

<details>

<summary>
Output shows result exceeds max_len by 7 characters
</summary>
```
Result: '9c9f57__0000__etc'
Length: 17
Expected max: 10
Violates constraint: True
```
</details>

## Why This Is A Bug

The function name `cap_length` with parameter `max_len` creates an explicit semantic contract that the returned string should never exceed `max_len` characters. However, the implementation fails this contract for `max_len < 17`.

The bug occurs in the format string construction at line 5708:
```python
return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```

The function creates:
- A 6-character hash prefix from SHA256
- Two underscores "__" (2 characters)
- A substring from the original string
- The suffix "__etc" (5 characters)

Total fixed overhead: 6 + 2 + 5 = 13 characters

When `max_len=10`, the slice `s[:max_len-17]` becomes `s[:-7]`, which for an 11-character string gives the first 4 characters. This results in a 17-character output ('9c9f57__0000__etc'), violating the 10-character maximum.

The function correctly handles strings that fit within max_len (lines 5705-5706), but the truncation logic assumes max_len is large enough to accommodate the fixed overhead plus at least some portion of the original string, which fails for small max_len values.

## Relevant Context

This internal function is used within the Cython compiler to create C identifiers with length constraints. The default `max_len=63` aligns with common C compiler identifier limits. Current usage within Cython (lines 3521, 5655, 5700 in PyrexTypes.py) all use the default or similarly large values, making this edge case unlikely in practice.

However, the bug represents a clear violation of the function's contract and could cause unexpected behavior if:
- Future code needs shorter identifier limits
- External code relies on this function
- Cython is used with compilers having stricter identifier limits

The function has no documentation or docstring explaining expected behavior or minimum max_len requirements.

## Proposed Fix

```diff
--- a/PyrexTypes.py
+++ b/PyrexTypes.py
@@ -5704,5 +5704,8 @@ def cap_length(s, max_len=63):
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+    if max_len < 17:
+        hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:max(4, max_len-5)]
+        return '%s__etc' % hash_prefix if max_len >= 9 else hash_prefix[:max_len]
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
     return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```