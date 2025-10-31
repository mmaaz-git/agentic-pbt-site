# Bug Report: scipy.io.arff.split_data_line IndexError on Empty String

**Target**: `scipy.io.arff._arffread.split_data_line`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `split_data_line` function in scipy.io.arff crashes with an IndexError when processing relational attributes that contain data ending with a newline character, causing the parser to pass an empty string to the function.

## Property-Based Test

```python
from io import StringIO

from hypothesis import given, settings, strategies as st
from scipy.io import arff


@given(
    has_trailing_newline=st.booleans(),
    num_rows=st.integers(min_value=1, max_value=10),
    values=st.data()
)
@settings(max_examples=200)
def test_relational_attribute_handles_trailing_newline(has_trailing_newline, num_rows, values):
    rows = []
    for _ in range(num_rows):
        val = values.draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
        rows.append(str(val))

    relational_data = '\\n'.join(rows)
    if has_trailing_newline:
        relational_data += '\\n'

    content = f"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"{relational_data}"
"""

    f = StringIO(content)
    data, meta = arff.loadarff(f)

    assert len(data) == 1

if __name__ == "__main__":
    test_relational_attribute_handles_trailing_newline()
```

<details>

<summary>
**Failing input**: `has_trailing_newline=True, num_rows=1, values containing 0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 38, in <module>
    test_relational_attribute_handles_trailing_newline()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 8, in test_relational_attribute_handles_trailing_newline
    has_trailing_newline=st.booleans(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 33, in test_relational_attribute_handles_trailing_newline
    data, meta = arff.loadarff(f)
                 ~~~~~~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    return _loadarff(ofile)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    a = list(generator(ofile))
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 867, in generator
    yield tuple([attr[i].parse_data(row[i]) for i in elems])
                 ~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 374, in parse_data
    row, self.dialect = split_data_line(raw, self.dialect)
                        ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
Falsifying example: test_relational_attribute_handles_trailing_newline(
    has_trailing_newline=True,
    num_rows=1,
    values=data(...),
)
Draw 1: 0.0
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/43/hypo.py:21
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from scipy.io import arff

content = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"0.0\n"
"""

f = StringIO(content)
data, meta = arff.loadarff(f)
```

<details>

<summary>
IndexError: string index out of range at line 476
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/repo.py", line 14, in <module>
    data, meta = arff.loadarff(f)
                 ~~~~~~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    return _loadarff(ofile)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    a = list(generator(ofile))
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 867, in generator
    yield tuple([attr[i].parse_data(row[i]) for i in elems])
                 ~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 374, in parse_data
    row, self.dialect = split_data_line(raw, self.dialect)
                        ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 476, in split_data_line
    if line[-1] == '\n':
       ~~~~^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

This is a clear programming error where `split_data_line` attempts to access `line[-1]` without first checking if the line is empty. The bug occurs when parsing relational attributes - an undocumented but implemented feature in SciPy's ARFF parser.

When relational data contains a trailing newline (e.g., `"0.0\n"`), the `RelationalAttribute.parse_data` method at line 373 splits the escaped string on newlines. Python's `split("\n")` on a string ending with `\n` produces an empty string as the last element. This empty string is then passed to `split_data_line`, which crashes at line 476 when trying to access `line[-1]`.

While relational attributes are not documented in SciPy's public API, the implementation exists and attempts to process them. Code that is accessible to users should not crash with low-level errors like IndexError, even for undocumented features. The proper behavior would be to either handle empty lines gracefully or raise a meaningful error message.

## Relevant Context

The ARFF format specification from Weka mentions relational attributes as "for multi-instance data (for future use)", suggesting this is an experimental feature. However, the specification doesn't explicitly define how trailing newlines within quoted relational data should be handled.

The bug affects any ARFF file with relational attributes where the data ends with a newline character - a common occurrence when data is generated programmatically or includes trailing whitespace.

The issue is in `/scipy/io/arff/_arffread.py`:
- Line 476: `split_data_line` tries to access `line[-1]` without checking if line is empty
- Line 374: `RelationalAttribute.parse_data` calls `split_data_line` for each line from splitting on `\n`

## Proposed Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -473,7 +473,7 @@ def split_data_line(line, dialect=None):
     csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

     # Remove the line end if any
-    if line[-1] == '\n':
+    if line and line[-1] == '\n':
         line = line[:-1]

     # Remove potential trailing whitespace
```