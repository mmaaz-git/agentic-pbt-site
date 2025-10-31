# Bug Report: Oracle Backend Timezone Regex Overly Permissive

**Target**: `django.db.backends.oracle.operations.DatabaseOperations._tzname_re`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The timezone name validation regex `^[\w/:+-]+$` accepts Unicode word characters (including Chinese, Greek, Cyrillic, etc.) even though all valid timezone names in the IANA timezone database use only ASCII characters. This is overly permissive and contradicts the code comment stating "This regexp matches all time zone names from the zoneinfo database."

## Property-Based Test

```python
from hypothesis import given, strategies as st
import re

_tzname_re = re.compile(r"^[\w/:+-]+$")

@given(st.text(min_size=1))
def test_tzname_regex_matches_only_valid_chars(tzname):
    if _tzname_re.match(tzname):
        # If accepted, should only contain ASCII word chars and /:-+
        assert all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/:+-" for c in tzname)
```

**Failing input**: `'¹'`, `'中'`, `'Ω'` (and many other Unicode characters)

## Reproducing the Bug

```python
import re

_tzname_re = re.compile(r"^[\w/:+-]+$")

test_cases = [
    ("UTC", "Standard timezone", True),
    ("America/New_York", "Standard timezone", True),
    ("¹", "Superscript 1 (Unicode)", False),
    ("中国/北京", "Chinese characters", False),
    ("Москва", "Cyrillic characters", False),
]

for tz, description, should_accept in test_cases:
    matches = _tzname_re.match(tz)
    expected = "should accept" if should_accept else "should reject"
    actual = "accepts" if matches else "rejects"

    print(f"{tz!r:20} ({description}): {expected} -> {actual}")
    if bool(matches) != should_accept:
        print(f"  ❌ BUG: Expected {expected} but regex {actual}")
```

**Output:**
```
'UTC'                (Standard timezone): should accept -> accepts
'America/New_York'   (Standard timezone): should accept -> accepts
'¹'                  (Superscript 1 (Unicode)): should reject -> accepts
  ❌ BUG: Expected should reject but regex accepts
'中国/北京'           (Chinese characters): should reject -> accepts
  ❌ BUG: Expected should reject but regex accepts
'Москва'             (Cyrillic characters): should reject -> accepts
  ❌ BUG: Expected should reject but regex accepts
```

## Why This Is A Bug

The regex uses `\w` which in Python matches ALL Unicode word characters, not just ASCII. According to the IANA timezone database, all valid timezone names use only:
- ASCII letters (a-z, A-Z)
- ASCII digits (0-9)
- Underscores (_)
- Slashes (/)
- Colons (:)
- Plus/minus (+/-)

The code comment explicitly states: "This regexp matches all time zone names from the zoneinfo database" but the regex actually matches far MORE than that.

**Impact:**
- Invalid timezone names pass validation and reach Oracle
- Oracle rejects them with confusing error messages
- Users don't get clear validation errors
- The overly permissive regex contradicts its documented purpose

**Not a security issue:** The regex correctly blocks SQL injection attempts (quotes, semicolons, parentheses are rejected), so this is purely a validation quality issue.

## Fix

Replace `\w` with explicit ASCII character class:

```diff
-    _tzname_re = _lazy_re_compile(r"^[\w/:+-]+$")
+    _tzname_re = _lazy_re_compile(r"^[a-zA-Z0-9_/:+-]+$")
```