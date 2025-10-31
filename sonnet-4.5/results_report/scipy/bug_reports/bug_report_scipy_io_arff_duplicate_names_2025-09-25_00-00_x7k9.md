# Bug Report: scipy.io.arff Duplicate Attribute Names

**Target**: `scipy.io.arff.loadarff`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.io.arff.loadarff()` crashes with a confusing ValueError when parsing ARFF files with duplicate attribute names, instead of raising a clear ParseArffError during header validation.

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

**Failing input**: `arff_components=('a', [('a', 'numeric'), ('a', 'numeric')], ['0.0,0.0'])`

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

**Output:**
```
ValueError: field 'width' occurs more than once
```

**Expected behavior**: Should raise `arff.ParseArffError` with a clear message like "Duplicate attribute name 'width' found in header".

## Why This Is A Bug

1. **Wrong exception type**: Raises `ValueError` from NumPy internals instead of `arff.ParseArffError`
2. **No early validation**: Duplicate attribute names should be caught during header parsing
3. **Confusing error message**: Users don't know the problem is duplicate names in their ARFF file
4. **Code comment confirms it**: Line 870 in `_arffread.py` says "No error should happen here: it is a bug otherwise"

The ARFF format specification doesn't explicitly allow duplicate attribute names (attribute names should uniquely identify columns). scipy should validate this during header parsing.

## Fix

Add validation in the `read_header` function to check for duplicate attribute names:

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -somewhere in read_header function
+    # Check for duplicate attribute names
+    attr_names = [a.name for a in attr]
+    duplicates = [name for name in attr_names if attr_names.count(name) > 1]
+    if duplicates:
+        raise ParseArffError(
+            f"Duplicate attribute name(s) found: {set(duplicates)}"
+        )
+
     return rel, attr
```

This ensures duplicate names are caught early with a clear, helpful error message using the appropriate exception type.