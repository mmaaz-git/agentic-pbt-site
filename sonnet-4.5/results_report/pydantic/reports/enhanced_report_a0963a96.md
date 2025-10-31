# Bug Report: pydantic.experimental.pipeline str_strip() Fails to Strip Unicode Control Characters U+001C-001F

**Target**: `pydantic.experimental.pipeline._Pipeline.str_strip`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `str_strip()` method in pydantic's experimental pipeline fails to strip Unicode control characters U+001C through U+001F (File Separator, Group Separator, Record Separator, Unit Separator), which Python's native `str.strip()` correctly removes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_strip_matches_python(text):
    class Model(BaseModel):
        value: Annotated[str, validate_as(str).str_strip()]

    m = Model(value=text)
    assert m.value == text.strip()

if __name__ == "__main__":
    test_str_strip_matches_python()
```

<details>

<summary>
**Failing input**: `text='\x1f'`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 16, in <module>
    test_str_strip_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_str_strip_matches_python
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 13, in test_str_strip_matches_python
    assert m.value == text.strip()
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_str_strip_matches_python(
    text='\x1f',
)
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, validate_as(str).str_strip()]

# Test with Unicode Unit Separator (U+001F)
m = Model(value='\x1f')
print(f"Result: {m.value!r}")
print(f"Expected: {'\x1f'.strip()!r}")
print(f"Are they equal? {m.value == '\x1f'.strip()}")

# Additional test with various Unicode whitespace
test_strings = [
    '\x1f',  # Unicode Unit Separator
    '\x1c',  # File Separator
    '\x1d',  # Group Separator
    '\x1e',  # Record Separator
    '\x0b',  # Vertical Tab
    '\x0c',  # Form Feed
    '\x85',  # Next Line
    '\xa0',  # Non-breaking space
    '\u2000',  # En Quad
    '\u2001',  # Em Quad
    '\u2002',  # En Space
    '\u2003',  # Em Space
    '\u2004',  # Three-Per-Em Space
    '\u2005',  # Four-Per-Em Space
    '\u2006',  # Six-Per-Em Space
    '\u2007',  # Figure Space
    '\u2008',  # Punctuation Space
    '\u2009',  # Thin Space
    '\u200a',  # Hair Space
    '\u202f',  # Narrow No-Break Space
    '\u205f',  # Medium Mathematical Space
    '\u3000',  # Ideographic Space
]

print("\n--- Testing various Unicode whitespace characters ---")
for test in test_strings:
    model = Model(value=test)
    expected = test.strip()
    print(f"Input: {test!r} (U+{ord(test):04X})")
    print(f"  pydantic result: {model.value!r}")
    print(f"  Python strip(): {expected!r}")
    print(f"  Match: {model.value == expected}")
    if model.value != expected:
        print(f"  ❌ MISMATCH!")
    print()
```

<details>

<summary>
Error: str_strip() doesn't strip Unicode control characters
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Result: '\x1f'
Expected: ''
Are they equal? False

--- Testing various Unicode whitespace characters ---
Input: '\x1f' (U+001F)
  pydantic result: '\x1f'
  Python strip(): ''
  Match: False
  ❌ MISMATCH!

Input: '\x1c' (U+001C)
  pydantic result: '\x1c'
  Python strip(): ''
  Match: False
  ❌ MISMATCH!

Input: '\x1d' (U+001D)
  pydantic result: '\x1d'
  Python strip(): ''
  Match: False
  ❌ MISMATCH!

Input: '\x1e' (U+001E)
  pydantic result: '\x1e'
  Python strip(): ''
  Match: False
  ❌ MISMATCH!

Input: '\x0b' (U+000B)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\x0c' (U+000C)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\x85' (U+0085)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\xa0' (U+00A0)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2000' (U+2000)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2001' (U+2001)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2002' (U+2002)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2003' (U+2003)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2004' (U+2004)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2005' (U+2005)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2006' (U+2006)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2007' (U+2007)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2008' (U+2008)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u2009' (U+2009)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u200a' (U+200A)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u202f' (U+202F)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u205f' (U+205F)
  pydantic result: ''
  Python strip(): ''
  Match: True

Input: '\u3000' (U+3000)
  pydantic result: ''
  Python strip(): ''
  Match: True

```
</details>

## Why This Is A Bug

The `str_strip()` method is explicitly implemented to use Python's `str.strip()` method (see line 310-311 of pipeline.py):

```python
def str_strip(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
    return self.transform(str.strip)
```

This clearly indicates that `str_strip()` should behave identically to Python's native `str.strip()` method. However, an optimization in the `_apply_transform` function (lines 428-431) replaces the `str.strip` transformation with pydantic-core's `strip_whitespace` flag:

```python
if func is str.strip:
    s = s.copy()
    s['strip_whitespace'] = True
    return s
```

The pydantic-core `strip_whitespace` flag strips most Unicode whitespace characters correctly (including spaces, tabs, newlines, and many Unicode space characters like U+2000-U+3000), but it fails to strip Unicode control characters U+001C through U+001F. These are legitimate whitespace characters according to Python's Unicode handling:

- U+001C (File Separator)
- U+001D (Group Separator)
- U+001E (Record Separator)
- U+001F (Unit Separator)

Python's `str.strip()` correctly identifies and removes these control characters as whitespace, while pydantic's optimized implementation does not, creating an inconsistency between the expected and actual behavior.

## Relevant Context

1. **Module Status**: The pydantic experimental pipeline module is marked as experimental and subject to change, as evidenced by the warning message.

2. **Performance vs Correctness Trade-off**: The optimization appears to be for performance reasons, but it sacrifices correctness for a subset of Unicode whitespace characters.

3. **Affected Characters**: The bug specifically affects Unicode control characters in the range U+001C to U+001F. These are rarely used in typical text processing but are still valid whitespace according to Unicode standards.

4. **Workaround Available**: Users can work around this issue by using `.transform(str.strip)` directly instead of `.str_strip()`, which bypasses the optimization.

5. **Related Code**: The issue is in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`

## Proposed Fix

Remove the optimization for `str.strip` to ensure correct behavior that matches Python's native `str.strip()`:

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -425,11 +425,7 @@ def _apply_transform(
         return cs.no_info_plain_validator_function(func)

     if s['type'] == 'str':
-        if func is str.strip:
-            s = s.copy()
-            s['strip_whitespace'] = True
-            return s
-        elif func is str.lower:
+        if func is str.lower:
             s = s.copy()
             s['to_lower'] = True
             return s
```