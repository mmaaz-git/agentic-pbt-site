# Bug Report: pandas CSV Round-Trip Data Corruption with Tab Character in Column Name

**Target**: `pandas.DataFrame.to_csv` / `pandas.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame column name contains only a tab character (`'\t'`), CSV round-trip with default settings silently corrupts data: the data value becomes the column name and all actual data is lost.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=10))
@settings(max_examples=200)
def test_csv_handles_special_chars_in_column_names(name):
    df = pd.DataFrame([[1]], columns=[name])
    csv_str = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_str))

    assert len(result.columns) == 1
    assert result.columns[0] == name

if __name__ == "__main__":
    test_csv_handles_special_chars_in_column_names()
```

<details>

<summary>
**Failing input**: `'\x00'` (null character found first, but `'\t'` also fails)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 16, in <module>
    test_csv_handles_special_chars_in_column_names()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 6, in test_csv_handles_special_chars_in_column_names
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 13, in test_csv_handles_special_chars_in_column_names
    assert result.columns[0] == name
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_csv_handles_special_chars_in_column_names(
    name='\x00',
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame([[42]], columns=['\t'])
print("Original:")
print(f"  Columns: {list(df.columns)}")
print(f"  Values: {df.values.tolist()}")

csv_str = df.to_csv(index=False)
print("\nCSV output:")
print(repr(csv_str))

result = pd.read_csv(io.StringIO(csv_str))

print("\nAfter round-trip:")
print(f"  Columns: {list(result.columns)}")
print(f"  Values: {result.values.tolist()}")
```

<details>

<summary>
Data corruption: value '42' becomes column name, data is lost
</summary>
```
Original:
  Columns: ['\t']
  Values: [[42]]

CSV output:
'\t\n42\n'

After round-trip:
  Columns: ['42']
  Values: []
```
</details>

## Why This Is A Bug

This violates the fundamental expectation of CSV round-trip preservation. The default `quoting=csv.QUOTE_MINIMAL` setting in `to_csv()` fails to quote column names containing delimiter characters (tabs, which are valid CSV delimiters). When `read_csv()` parses the unquoted tab character in the header row, it interprets it as a field separator rather than a column name, leading to:

1. **Silent data corruption**: No error or warning is raised
2. **Complete data loss**: The DataFrame becomes empty
3. **Column/data confusion**: The data value (42) incorrectly becomes the column name
4. **Lost metadata**: The original column name is completely lost

The pandas documentation states that `to_csv()` writes DataFrames to CSV format and `read_csv()` reads them back, implying round-trip compatibility. This expectation is violated when column names contain characters that have special meaning in CSV format but are not properly escaped with the default settings.

## Relevant Context

Testing revealed that this issue affects multiple special characters:
- Tab character (`'\t'`) causes the reported data corruption
- Null character (`'\x00'`) results in column name becoming "Unnamed: 0"
- The issue only occurs with `quoting=csv.QUOTE_MINIMAL` (default) and `quoting=csv.QUOTE_NONE`
- Using `quoting=csv.QUOTE_ALL` or `quoting=csv.QUOTE_NONNUMERIC` properly preserves the data

The CSV format treats tabs as valid field delimiters. When pandas writes a tab character as a column name without quoting, it creates ambiguous CSV that cannot be correctly parsed on read.

Relevant pandas documentation:
- [DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
- [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

## Proposed Fix

The `to_csv` method should automatically detect and quote column names containing delimiter characters when using `QUOTE_MINIMAL` mode. Here's a high-level approach:

1. In the CSV writer logic, check if any column name contains characters that could be interpreted as delimiters (tab, comma, newline, etc.)
2. If such characters are detected, force quoting for those column names even in QUOTE_MINIMAL mode
3. Alternatively, issue a warning when column names contain delimiter characters and suggest using QUOTE_ALL

A temporary workaround for users is to explicitly use `quoting=csv.QUOTE_ALL` when dealing with DataFrames that might have special characters in column names:

```python
df.to_csv('file.csv', index=False, quoting=csv.QUOTE_ALL)
```