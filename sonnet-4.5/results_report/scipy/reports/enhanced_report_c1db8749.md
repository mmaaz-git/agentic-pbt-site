# Bug Report: scipy.io.arff.loadarff Duplicate Attribute Names Crash

**Target**: `scipy.io.arff.loadarff`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.arff.loadarff()` crashes with a confusing ValueError from NumPy internals when parsing ARFF files with duplicate attribute names, instead of raising a ParseArffError during header validation as expected.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io import arff
from io import StringIO


@st.composite
def valid_arff_identifier(draw):
    first_char = draw(st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', max_size=20))
    return first_char + rest


@st.composite
def arff_mixed_attributes(draw):
    num_attrs = draw(st.integers(min_value=1, max_value=8))
    attributes = []
    for i in range(num_attrs):
        attr_name = draw(valid_arff_identifier())
        is_numeric = draw(st.booleans())
        if is_numeric:
            attributes.append((attr_name, 'numeric'))
        else:
            num_values = draw(st.integers(min_value=2, max_value=5))
            nominal_values = [draw(valid_arff_identifier()) for _ in range(num_values)]
            attributes.append((attr_name, tuple(set(nominal_values))))
    return attributes


def generate_arff_content(relation_name, attributes, data_rows):
    lines = [f"@relation {relation_name}"]
    for attr_name, attr_type in attributes:
        if isinstance(attr_type, str):
            lines.append(f"@attribute {attr_name} {attr_type}")
        else:
            nominal_values = ','.join(attr_type)
            lines.append(f"@attribute {attr_name} {{{nominal_values}}}")
    lines.append("@data")
    lines.extend(data_rows)
    return '\n'.join(lines)


@settings(max_examples=200)
@given(
    relation_name=valid_arff_identifier(),
    attributes=arff_mixed_attributes(),
    data_rows=st.lists(st.text(), min_size=1, max_size=20)
)
def test_loadarff_handles_all_valid_inputs(relation_name, attributes, data_rows):
    content = generate_arff_content(relation_name, attributes, data_rows)
    f = StringIO(content)
    data, meta = arff.loadarff(f)
```

<details>

<summary>
**Failing input**: `attributes=[('a', 'numeric'), ('a', 'numeric')], data_rows=['']`
</summary>
```
+ Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 55, in <module>
  |     test_loadarff_handles_all_valid_inputs()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 43, in test_loadarff_handles_all_valid_inputs
  |     @given(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 5 distinct failures. (5 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 51, in test_loadarff_handles_all_valid_inputs
    |     data, meta = arff.loadarff(f)
    |                  ~~~~~~~~~~~~~^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    |     return _loadarff(ofile)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    |     a = list(generator(ofile))
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 867, in generator
    |     yield tuple([attr[i].parse_data(row[i]) for i in elems])
    |                                     ~~~^^^
    | IndexError: list index out of range
    | Falsifying example: test_loadarff_handles_all_valid_inputs(
    |     relation_name='a',  # or any other generated value
    |     attributes=[('a', 'numeric'), ('a', ('a',))],
    |     data_rows=['0'],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 51, in test_loadarff_handles_all_valid_inputs
    |     data, meta = arff.loadarff(f)
    |                  ~~~~~~~~~~~~~^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    |     return _loadarff(ofile)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    |     a = list(generator(ofile))
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 867, in generator
    |     yield tuple([attr[i].parse_data(row[i]) for i in elems])
    |                  ~~~~~~~~~~~~~~~~~~^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 159, in parse_data
    |     raise ValueError(f"{str(data_str)} value not in {str(self.values)}")
    | ValueError: 0 value not in ('a',)
    | Falsifying example: test_loadarff_handles_all_valid_inputs(
    |     relation_name='a',  # or any other generated value
    |     attributes=[('a', ('a',))],
    |     data_rows=['0'],
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py:154
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 51, in test_loadarff_handles_all_valid_inputs
    |     data, meta = arff.loadarff(f)
    |                  ~~~~~~~~~~~~~^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    |     return _loadarff(ofile)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 871, in _loadarff
    |     data = np.array(a, [(a.name, a.dtype) for a in attr])
    | ValueError: field 'a' occurs more than once
    | Falsifying example: test_loadarff_handles_all_valid_inputs(
    |     relation_name='a',  # or any other generated value
    |     attributes=[('a', 'numeric'), ('a', 'numeric')],
    |     data_rows=[''],
    | )
    +---------------- 4 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 51, in test_loadarff_handles_all_valid_inputs
    |     data, meta = arff.loadarff(f)
    |                  ~~~~~~~~~~~~~^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    |     return _loadarff(ofile)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    |     a = list(generator(ofile))
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 865, in generator
    |     row, dialect = split_data_line(raw, dialect)
    |                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 495, in split_data_line
    |     row = next(csv.reader([line], dialect))
    | _csv.Error: new-line character seen in unquoted field - do you need to open the file with newline=''?
    | Falsifying example: test_loadarff_handles_all_valid_inputs(
    |     relation_name='a',
    |     attributes=[('b0', 'numeric')],
    |     data_rows=['0\r0'],
    | )
    +---------------- 5 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 51, in test_loadarff_handles_all_valid_inputs
    |     data, meta = arff.loadarff(f)
    |                  ~~~~~~~~~~~~~^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    |     return _loadarff(ofile)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 869, in _loadarff
    |     a = list(generator(ofile))
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 867, in generator
    |     yield tuple([attr[i].parse_data(row[i]) for i in elems])
    |                  ~~~~~~~~~~~~~~~~~~^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 224, in parse_data
    |     return float(data_str)
    | ValueError: could not convert string to float: ':'
    | Falsifying example: test_loadarff_handles_all_valid_inputs(
    |     relation_name='a',  # or any other generated value
    |     attributes=[('a', 'numeric')],  # or any other generated value
    |     data_rows=[':'],
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from scipy.io import arff

arff_content = """@relation test
@attribute width numeric
@attribute width numeric
@data
5.0,3.25
4.5,3.75
"""

f = StringIO(arff_content)
data, meta = arff.loadarff(f)
```

<details>

<summary>
ValueError: field 'width' occurs more than once
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/repo.py", line 13, in <module>
    data, meta = arff.loadarff(f)
                 ~~~~~~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 804, in loadarff
    return _loadarff(ofile)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py", line 871, in _loadarff
    data = np.array(a, [(a.name, a.dtype) for a in attr])
ValueError: field 'width' occurs more than once
```
</details>

## Why This Is A Bug

1. **Wrong exception type**: The function raises `ValueError` from NumPy's internal array creation code instead of `ParseArffError` which is documented as the exception type for invalid ARFF files.

2. **Violates code comment contract**: Line 870 in `_arffread.py` explicitly states: "# No error should happen here: it is a bug otherwise". The very next line (871) is where this error occurs, confirming this is indeed a bug by the code's own admission.

3. **No early validation**: According to the ARFF specification used by Weka, attribute names must be unique. The parser should validate this constraint during header parsing, not allow invalid data to propagate to NumPy array creation.

4. **Confusing error message**: Users receive "field 'width' occurs more than once" from NumPy internals, which doesn't clearly indicate that the problem is duplicate attribute names in their ARFF file header.

5. **Documentation mismatch**: The function's docstring states it raises `ParseArffError` for invalid ARFF files and `NotImplementedError` for unsupported features. `ValueError` is not documented as a possible exception.

## Relevant Context

The ARFF format specification from Weka clearly states that each attribute has its own @attribute statement which "uniquely defines the name of that attribute". When converting CSV files to ARFF, Weka itself raises "IllegalArgumentException: Attribute names are not unique" for duplicate names.

The bug appears at line 871 in `/home/npc/.local/lib/python3.13/site-packages/scipy/io/arff/_arffread.py`:
```python
# No error should happen here: it is a bug otherwise
data = np.array(a, [(a.name, a.dtype) for a in attr])
```

This creates a NumPy structured array where field names must be unique, but no validation occurs beforehand during the `read_header()` function at line 624-652.

## Proposed Fix

Add duplicate attribute name validation in the `read_header` function after collecting all attributes:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -650,6 +650,13 @@ def read_header(ofile):
         else:
             i = next(ofile)

+    # Check for duplicate attribute names
+    attr_names = [attr.name for attr in attributes]
+    seen = set()
+    duplicates = [name for name in attr_names if name in seen or seen.add(name)]
+    if duplicates:
+        raise ParseArffError(f"Duplicate attribute name(s) found: {', '.join(sorted(set(duplicates)))}")
+
     return relation, attributes
```