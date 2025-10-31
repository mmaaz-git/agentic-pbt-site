# Bug Report: pandas.io.parsers.read_csv Regex Special Characters in Separators

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using multi-character separators containing regex special characters (like `||`, `..`, `++`, etc.) with `read_csv()` and the Python engine, pandas incorrectly interprets them as regex patterns instead of literal strings, causing parsing errors or incorrect results.

## Property-Based Test

```python
import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st, settings
import pytest
import re

regex_special_chars = ['|', '+', '*', '?', '.', '^', '$']

@given(
    special_char=st.sampled_from(regex_special_chars),
    num_cols=st.integers(min_value=2, max_value=5),
    num_rows=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_regex_special_char_separators(special_char, num_cols, num_rows):
    separator = special_char * 2

    header = separator.join([f'col{i}' for i in range(num_cols)])
    rows = []
    for i in range(num_rows):
        row = separator.join([str(i * num_cols + j) for j in range(num_cols)])
        rows.append(row)

    csv_content = header + '\n' + '\n'.join(rows)

    try:
        df = pd.read_csv(StringIO(csv_content), sep=separator, engine='python')

        if df.shape[1] != num_cols:
            expected_cols = [f'col{i}' for i in range(num_cols)]
            actual_cols = list(df.columns)
            pytest.fail(
                f"Separator {repr(separator)} failed: expected {num_cols} columns {expected_cols}, "
                f"got {df.shape[1]} columns {actual_cols}"
            )
    except (pd.errors.ParserError, ValueError, re.error) as e:
        pass
```

**Failing inputs**:
- `special_char='|'` (separator `||`)
- `special_char='.'` (separator `..`)
- `special_char='+'` (separator `++`) - causes `PatternError`
- `special_char='*'` (separator `**`) - causes `PatternError`
- `special_char='?'` (separator `??`) - causes `PatternError`

## Reproducing the Bug

### Case 1: `||` separator (wrong column count)
```python
import pandas as pd
from io import StringIO

csv_data = 'col0||col1\n0||1'
df = pd.read_csv(StringIO(csv_data), sep='||', engine='python')

print(f"Expected: 2 columns ['col0', 'col1']")
print(f"Got: {df.shape[1]} columns {list(df.columns)}")
```

**Output:**
```
Expected: 2 columns ['col0', 'col1']
Got: 12 columns ['Unnamed: 0', 'c', 'o', 'l', '0', '|', '|.1', 'c.1', 'o.1', 'l.1', '1', 'Unnamed: 11']
```

### Case 2: `..` separator (wrong column count)
```python
csv_data = 'col0..col1\n0..1'
df = pd.read_csv(StringIO(csv_data), sep='..', engine='python')

print(f"Expected: 2 columns")
print(f"Got: {df.shape[1]} columns")
```

**Output:**
```
Expected: 2 columns
Got: 6 columns
```

### Case 3: `++` separator (regex error)
```python
csv_data = 'col0++col1\n0++1'
df = pd.read_csv(StringIO(csv_data), sep='++', engine='python')
```

**Output:**
```
re.PatternError: nothing to repeat at position 0
```

## Why This Is A Bug

According to pandas documentation, multi-character separators are treated as regular expressions. However, when users specify separators like `||`, `..`, or `++`, they almost certainly intend them as literal strings, not regex patterns.

**Root causes:**
1. **`||`**: In regex, `|` is the alternation operator. The pattern `||` means "empty OR empty", which matches at every position, causing splits on individual characters.
2. **`..`**: In regex, `.` matches any character. The pattern `..` matches any two characters, causing incorrect parsing.
3. **`++`, `**`, `??`**: These are regex quantifiers that require a preceding element, causing `PatternError`.

This violates the principle of least surprise and makes pandas unusable for files with these common delimiter choices without knowing the workaround.

**Workaround:** Users must manually escape the separator using `re.escape()`:
```python
import re
df = pd.read_csv(StringIO(csv_data), sep=re.escape('||'), engine='python')
```

Or use explicit escaping:
```python
df = pd.read_csv(StringIO(csv_data), sep=r'\|\|', engine='python')
```

## Fix

Pandas should automatically escape special regex characters when the separator contains regex metacharacters, or at minimum, detect likely mistakes and warn users.

**Option A (Auto-escape):** When `sep` contains regex metacharacters but looks like it's intended to be literal, automatically escape it.

**Option B (Validation):** Detect separators that are likely to be mistakes and raise a clear error message.

**Option C (Documentation):** At minimum, improve documentation with explicit warnings and examples.

Recommended implementation (Option B):

```diff
--- a/pandas/io/parsers/python_parser.py
+++ b/pandas/io/parsers/python_parser.py
@@ -93,6 +93,18 @@ class PythonParser(ParserBase):
         self.delimiter = kwds["delimiter"]
+
+        if self.delimiter and len(self.delimiter) > 1:
+            import re
+            try:
+                if re.match(self.delimiter, ""):
+                    raise ValueError(
+                        f"The separator pattern {repr(self.delimiter)} matches empty strings. "
+                        f"Did you mean {repr(re.escape(self.delimiter))}?"
+                    )
+            except re.error as e:
+                raise ValueError(
+                    f"Invalid regex pattern in separator {repr(self.delimiter)}: {e}. "
+                    f"For literal separators, use {repr(re.escape(self.delimiter))}"
+                )

         self.quotechar = kwds["quotechar"]
         if isinstance(self.quotechar, str):
```