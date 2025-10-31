# Bug Report: Cython.Utils.build_hex_version ValueError on Invalid Input

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `build_hex_version` function crashes with an uninformative `ValueError` when given invalid version strings such as empty strings, non-numeric inputs, or improperly formatted versions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import build_hex_version

@given(st.text())
def test_build_hex_version_handles_all_strings(version_str):
    try:
        result = build_hex_version(version_str)
        assert result.startswith('0x')
        assert len(result) == 10
    except ValueError as e:
        assert "invalid literal for int()" not in str(e), \
            f"Should raise informative error, not '{e}'"
```

**Failing input**: `""` (empty string)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/cython')

from Cython.Utils import build_hex_version

print(build_hex_version(""))
```

**Output**:
```
Traceback (most recent call last):
  File "test.py", line 5, in <module>
    print(build_hex_version(""))
  File ".../Cython/Utils.py", line 611, in build_hex_version
    digits.append(int(segment))
ValueError: invalid literal for int() with base 10: ''
```

**Additional failing inputs**:
- `build_hex_version(".")` - just a dot
- `build_hex_version("a")` - alphabetic character
- `build_hex_version("1.0foo")` - version with invalid suffix
- `build_hex_version("1..0")` - double dot

## Why This Is A Bug

The function's docstring states it parses "public version identifier like '4.3a1'" according to PEP 440, but it doesn't validate input or provide helpful error messages for invalid formats. When `re.split(r'(\D+)', version_string)` produces empty strings (which happens with empty input, leading/trailing delimiters, or consecutive delimiters), the code attempts `int('')` which raises a cryptic ValueError.

Users calling this function with potentially invalid input (e.g., from user input or external data) receive an unhelpful error message that doesn't indicate the version string format is invalid.

## Fix

Add input validation at the start of the function to check for empty or invalid segments:

```diff
def build_hex_version(version_string):
    """
    Parse and translate public version identifier like '4.3a1' into the readable hex representation '0x040300A1' (like PY_VERSION_HEX).

    SEE: https://peps.python.org/pep-0440/#public-version-identifiers
    """
+   if not version_string or not version_string[0].isdigit():
+       raise ValueError(f"Invalid version string: {version_string!r}. Must start with a digit.")
+
    # Parse '4.12a1' into [4, 12, 0, 0xA01]
    # And ignore .dev, .pre and .post segments
    digits = []
    release_status = 0xF0
    for segment in re.split(r'(\D+)', version_string):
        if segment in ('a', 'b', 'rc'):
            release_status = {'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}[segment]
            digits = (digits + [0, 0])[:3]  # 1.2a1 -> 1.2.0a1
        elif segment in ('.dev', '.pre', '.post'):
            break  # break since those are the last segments
        elif segment != '.':
+           if not segment:
+               continue  # Skip empty segments from split
+           try:
+               digit = int(segment)
+           except ValueError:
+               raise ValueError(f"Invalid version string: {version_string!r}. "
+                              f"Segment {segment!r} is not a valid number.")
-           digits.append(int(segment))
+           digits.append(digit)

    digits = (digits + [0] * 3)[:4]
    digits[3] += release_status

    # Then, build a single hex value, two hex digits per version part.
    hexversion = 0
    for digit in digits:
        hexversion = (hexversion << 8) + digit

    return '0x%08X' % hexversion
```