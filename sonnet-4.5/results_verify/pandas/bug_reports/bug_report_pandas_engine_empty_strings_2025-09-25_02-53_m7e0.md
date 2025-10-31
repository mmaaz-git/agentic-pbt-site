# Bug Report: pandas.read_csv Engine Inconsistency with Empty Strings

**Target**: `pandas.io.parsers.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The C and Python parsing engines produce different results when reading CSV files containing quoted empty strings. The C engine treats them as NaN values, while the Python engine (with default skip_blank_lines=True) skips these rows entirely, resulting in different DataFrame shapes and data.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(
    text_data=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=['Cs', 'Cc']), min_size=0, max_size=20),
        min_size=1,
        max_size=10
    ),
    num_cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=200)
def test_engine_equivalence_text(text_data, num_cols):
    columns = [f'col{i}' for i in range(num_cols)]
    data = {col: text_data for col in columns}
    df = pd.DataFrame(data)
    csv_str = df.to_csv(index=False)

    df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
    df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)
```

**Failing input**: `text_data=['']` (list with single empty string)

## Reproducing the Bug

```python
import pandas as pd
import io

csv_str = 'col0\n""\na\n""\n'

df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

print("C engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")

print("\nPython engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")
```

Output:
```
C engine result:
  col0
0  NaN
1    a
2  NaN
Shape: (3, 1)

Python engine result:
  col0
0    a
Shape: (1, 1)
```

## Why This Is A Bug

This violates a fundamental invariant: **different parsing engines should produce identical results for the same input**. Users rely on engine selection for performance or compatibility reasons, not for different parsing semantics.

The specific issues:
1. **Data loss**: Python engine silently drops rows containing empty strings
2. **Shape mismatch**: DataFrames have different numbers of rows (3 vs 1)
3. **Semantic difference**: C engine interprets `""` as NaN, Python engine treats it as a blank line to skip
4. **Quoted values ignored**: The CSV contains properly quoted empty strings `""`, which are distinct from blank lines

This bug affects real-world usage where:
- Empty strings are legitimate data values
- Users switch engines for performance without expecting behavioral changes
- Data integrity depends on preserving all rows

Note: When `skip_blank_lines=False` is explicitly set, both engines behave consistently, but the default behavior (`skip_blank_lines=True`) exposes the bug.

## Fix

The Python parser's blank line detection should not treat quoted empty strings as blank lines. A quoted empty string `""` is a valid field value and should be preserved, not skipped.

The fix should be in the Python parser's line skipping logic to:
1. Only skip lines that are truly blank (contain no data between delimiters)
2. Preserve lines with quoted empty strings as valid data rows
3. Match the C parser's interpretation of empty strings as NaN values

Alternatively, both parsers should be updated to:
- Distinguish between blank lines (no content) and empty field values (`""`)
- Handle `skip_blank_lines` consistently across engines