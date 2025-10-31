# Bug Report: Cython.Compiler.PyrexTypes.cap_length Returns Strings Exceeding max_len Parameter

**Target**: `Cython.Compiler.PyrexTypes.cap_length`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `cap_length()` function returns strings that exceed the specified `max_len` parameter when `max_len < 13`, violating its implied contract to cap string length at the specified maximum.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Compiler.PyrexTypes import cap_length


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'), st.integers(min_value=0, max_value=200))
@settings(max_examples=500)
def test_cap_length_honors_max_len(s, max_len):
    result = cap_length(s, max_len)
    assert len(result) <= max_len, f"cap_length({repr(s)}, {max_len}) returned {repr(result)} with length {len(result)}, exceeding max_len={max_len}"

if __name__ == "__main__":
    test_cap_length_honors_max_len()
```

<details>

<summary>
**Failing input**: `s='0'`, `max_len=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 12, in <module>
    test_cap_length_honors_max_len()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 6, in test_cap_length_honors_max_len
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 9, in test_cap_length_honors_max_len
    assert len(result) <= max_len, f"cap_length({repr(s)}, {max_len}) returned {repr(result)} with length {len(result)}, exceeding max_len={max_len}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: cap_length('0', 0) returned '5feceb____etc' with length 13, exceeding max_len=0
Falsifying example: test_cap_length_honors_max_len(
    s='0',
    max_len=0,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Compiler.PyrexTypes import cap_length

# Test with the minimal failing case from the report
result = cap_length('0', max_len=0)
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
print(f"Expected max length: 0")
print()

# Test with a few more cases to show the pattern
for max_len in [0, 5, 10, 12]:
    test_string = 'a' * 20  # A string longer than max_len
    result = cap_length(test_string, max_len=max_len)
    print(f"cap_length('{test_string[:10]}...', max_len={max_len})")
    print(f"  Result: {repr(result)}")
    print(f"  Result length: {len(result)} (expected <= {max_len})")
    if len(result) > max_len:
        print(f"  VIOLATION: Result exceeds max_len by {len(result) - max_len} characters")
    print()
```

<details>

<summary>
Output showing violations of max_len constraint for small values
</summary>
```
Result: '5feceb____etc'
Result length: 13
Expected max length: 0

cap_length('aaaaaaaaaa...', max_len=0)
  Result: '42492d__aaa__etc'
  Result length: 16 (expected <= 0)
  VIOLATION: Result exceeds max_len by 16 characters

cap_length('aaaaaaaaaa...', max_len=5)
  Result: '42492d__aaaaaaaa__etc'
  Result length: 21 (expected <= 5)
  VIOLATION: Result exceeds max_len by 16 characters

cap_length('aaaaaaaaaa...', max_len=10)
  Result: '42492d__aaaaaaaaaaaaa__etc'
  Result length: 26 (expected <= 10)
  VIOLATION: Result exceeds max_len by 16 characters

cap_length('aaaaaaaaaa...', max_len=12)
  Result: '42492d__aaaaaaaaaaaaaaa__etc'
  Result length: 28 (expected <= 12)
  VIOLATION: Result exceeds max_len by 16 characters

```
</details>

## Why This Is A Bug

The function `cap_length(s, max_len=63)` has a parameter named `max_len` which clearly indicates the maximum allowed length for the returned string. The function name itself, "cap_length", reinforces this contract - it should cap/limit the length of the input string to at most `max_len` characters.

However, when the input string `s` exceeds `max_len`, the function constructs a result using the format `'%s__%s__etc' % (hash_prefix, s[:max_len-17])`, where:
- `hash_prefix` is always 6 characters (from SHA256 hexdigest)
- The format adds "__" (2 chars), the truncated string, and "__etc" (5 chars)
- This creates a minimum possible length of 13 characters (6 + 2 + 0 + 5)

When `max_len < 13`, the slice `s[:max_len-17]` produces a negative index, which in Python results in taking characters from the end of the string rather than producing an empty slice. Even worse, for very small `max_len` values, this can actually make the result longer than intended. The function therefore always returns strings of at least 13 characters when truncation occurs, directly violating the contract implied by its name and parameter.

While the function lacks formal documentation, the parameter name `max_len` creates an unambiguous expectation based on standard programming conventions. A function that accepts a `max_len` parameter should honor it for all valid integer values, not just those above an arbitrary threshold.

## Relevant Context

The `cap_length` function is used internally in Cython's compiler to generate C identifier names while avoiding compiler limitations on identifier length. All three current uses in the codebase rely on the default `max_len=63`:

- Line 3521: `cap_length("_".join(arg_names))` - for function argument names
- Line 5655: `cap_length('__and_'.join(...))` - for type identifiers
- Line 5700: `cap_length(re.sub(...))` - for sanitized type identifiers

The function appears in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/PyrexTypes.py` at line 5704.

Since all current usage uses the default `max_len=63`, this bug has no impact on existing Cython functionality. However, the function is part of the public module interface and could be used by external code or future Cython features with different `max_len` values.

## Proposed Fix

```diff
--- a/Cython/Compiler/PyrexTypes.py
+++ b/Cython/Compiler/PyrexTypes.py
@@ -5704,6 +5704,9 @@ def type_identifier(type, pyrex=False):
 def cap_length(s, max_len=63):
     if len(s) <= max_len:
         return s
+    # Minimum format is "HASH__X__etc" (13 chars). For smaller max_len, just truncate.
+    if max_len < 13:
+        return s[:max_len]
     hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
     return '%s__%s__etc' % (hash_prefix, s[:max_len-17])
```