# Bug Report: pandas.read_csv Engine Inconsistency with Quoted Empty Strings

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The C and Python parsing engines in pandas.read_csv produce different DataFrame shapes and data when reading CSV files containing quoted empty strings (""), with the Python engine incorrectly dropping these rows as blank lines while the C engine correctly preserves them as NaN values.

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

if __name__ == "__main__":
    # Run the test
    test_engine_equivalence_text()
```

<details>

<summary>
**Failing input**: `text_data=[''], num_cols=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 27, in <module>
    test_engine_equivalence_text()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 6, in test_engine_equivalence_text
    text_data=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 23, in test_engine_equivalence_text
    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1242, in assert_frame_equal
    raise_assert_detail(
    ~~~~~~~~~~~~~~~~~~~^
        obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    raise AssertionError(msg)
AssertionError: DataFrame are different

DataFrame shape mismatch
[left]:  (1, 1)
[right]: (0, 1)
Falsifying example: test_engine_equivalence_text(
    text_data=[''],
    num_cols=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py:1242
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/range.py:557
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/construction.py:494
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/construction.py:606
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/internals/construction.py:611
        (and 13 more with settings.verbosity >= verbose)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import io

# Create a CSV string with quoted empty strings
csv_str = 'col0\n""\na\n""\n'

# Parse with C engine
df_c = pd.read_csv(io.StringIO(csv_str), engine='c')

# Parse with Python engine
df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

print("CSV input string:")
print(repr(csv_str))
print()

print("C engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")
print(f"Values: {df_c['col0'].tolist()}")
print()

print("Python engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")
print(f"Values: {df_python['col0'].tolist()}")
print()

print("Are the DataFrames equal?")
try:
    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)
    print("Yes, they are equal")
except AssertionError as e:
    print(f"No, they differ: {str(e)}")
```

<details>

<summary>
DataFrame shape mismatch between C and Python engines
</summary>
```
CSV input string:
'col0\n""\na\n""\n'

C engine result:
  col0
0  NaN
1    a
2  NaN
Shape: (3, 1)
Values: [nan, 'a', nan]

Python engine result:
  col0
0    a
Shape: (1, 1)
Values: ['a']

Are the DataFrames equal?
No, they differ: DataFrame are different

DataFrame shape mismatch
[left]:  (3, 1)
[right]: (1, 1)
```
</details>

## Why This Is A Bug

This bug violates a fundamental expectation that different parsing engines should produce identical results for the same CSV input. The inconsistency arises from how each engine interprets quoted empty strings ("") when `skip_blank_lines=True` (the default setting).

According to CSV format standards (RFC 4180), a quoted empty string ("") represents an empty field value, not a blank line. A blank line in CSV would be a line containing no data between delimiters. The C engine correctly distinguishes between these two cases, treating quoted empty strings as valid empty field values (which become NaN in pandas), while the Python engine incorrectly treats them as blank lines to be skipped.

The specific problems are:

1. **Silent data loss**: The Python engine drops entire rows containing quoted empty strings without warning, potentially losing important data records.

2. **Shape inconsistency**: The same CSV input produces DataFrames with different numbers of rows (3 vs 1 in our example), making results unpredictable when switching engines.

3. **Semantic violation**: The Python engine's behavior contradicts CSV standards by conflating quoted empty fields with blank lines.

4. **Documentation ambiguity**: The pandas documentation does not clearly define what constitutes a "blank line" versus an "empty field," nor does it warn users about potential behavioral differences between engines.

This issue is particularly problematic because:
- Users typically choose between engines for performance reasons, not expecting different parsing semantics
- Empty strings are legitimate data values in many real-world datasets
- The data loss occurs silently without any warning or error
- Code that works correctly with one engine may produce incorrect results with another

## Relevant Context

Testing with pandas version 2.3.2 confirms this behavior. Interestingly, when `skip_blank_lines=False` is explicitly set, both engines produce identical results (3 rows with NaN, 'a', NaN), proving that both engines can handle the data correctly. This suggests the bug is specifically in the Python engine's implementation of blank line detection when `skip_blank_lines=True`.

The pandas documentation for the `engine` parameter describes the C engine as "faster" and the Python engine as "more feature-complete," implying they should be functionally equivalent for common operations. The documentation for `skip_blank_lines` states it will "skip over blank lines rather than interpreting as NaN values" but doesn't define what constitutes a blank line in the context of quoted fields.

Related pandas GitHub issue #21131 from 2018 documented similar engine inconsistencies with empty fields, which was apparently resolved, suggesting the pandas maintainers consider engine parity an important goal.

## Proposed Fix

The Python parser's blank line detection logic needs to be updated to distinguish between truly blank lines (no content) and lines containing quoted empty fields. The fix should be in the Python engine's line parsing code to match the C engine's behavior:

```diff
# Pseudocode for the fix in the Python engine's line skipping logic
# Location: pandas/io/parsers/python_parser.py (approximate)

def _is_blank_line(line):
-    # Current logic likely treats any line resulting in empty fields as blank
-    return all(field.strip() == '' for field in line)
+    # Fixed logic should preserve quoted empty strings
+    # Only skip lines that are truly blank (no content between delimiters)
+    if line.strip() == '':
+        return True  # Truly blank line
+    # Check if line contains quoted empty strings
+    if '""' in line:
+        return False  # Quoted empty string is a valid field
+    # Otherwise check if all fields are empty
+    return all(field.strip() == '' for field in parse_line(line))
```

A more robust fix would involve properly parsing the CSV line to distinguish between:
1. Blank lines: `\n` or lines with only whitespace
2. Lines with empty unquoted fields: `,,\n`
3. Lines with quoted empty fields: `"","",""\n`

The Python engine should only skip case 1 when `skip_blank_lines=True`, while preserving cases 2 and 3 as valid data rows with NaN values, matching the C engine's behavior.