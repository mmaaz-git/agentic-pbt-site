# Bug Report: scipy.io.arff Inconsistent Quote Stripping

**Target**: `scipy.io.arff.loadarff`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The ARFF parser inconsistently strips quotes from attribute names: double quotes are never stripped, and single quotes are only stripped for names with 2+ characters, but not for single-character names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.io import arff
from io import StringIO


@settings(max_examples=100)
@given(
    attr_name=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    )
)
def test_quote_stripping_consistency(attr_name):
    single_quote_arff = f"""
@relation test
@attribute '{attr_name}' numeric
@data
1.0
"""

    double_quote_arff = f"""
@relation test
@attribute "{attr_name}" numeric
@data
1.0
"""

    f1 = StringIO(single_quote_arff)
    data1, meta1 = arff.loadarff(f1)

    f2 = StringIO(double_quote_arff)
    data2, meta2 = arff.loadarff(f2)

    name_with_single_quotes = meta1.names()[0]
    name_with_double_quotes = meta2.names()[0]

    assert name_with_single_quotes == attr_name
    assert name_with_double_quotes == attr_name
    assert name_with_single_quotes == name_with_double_quotes
```

**Failing input**: `attr_name='a'` (or any attribute name)

## Reproducing the Bug

```python
from scipy.io import arff
from io import StringIO

arff1 = "@relation test\n@attribute 'a' numeric\n@data\n1.0"
data1, meta1 = arff.loadarff(StringIO(arff1))
print(meta1.names()[0])

arff2 = "@relation test\n@attribute 'ab' numeric\n@data\n1.0"
data2, meta2 = arff.loadarff(StringIO(arff2))
print(meta2.names()[0])

arff3 = "@relation test\n@attribute \"myattr\" numeric\n@data\n1.0"
data3, meta3 = arff.loadarff(StringIO(arff3))
print(meta3.names()[0])
```

Expected output:
```
a
ab
myattr
```

Actual output:
```
'a'
ab
"myattr"
```

## Why This Is A Bug

According to ARFF format specification, quotes (both single and double) are used to allow special characters and spaces in attribute names, and should be stripped during parsing. The current implementation:

1. **Never strips double quotes**: `"myattr"` → `"myattr"` (includes quotes in name)
2. **Inconsistently strips single quotes**:
   - Single-char names: `'a'` → `'a'` (quotes retained)
   - Multi-char names: `'ab'` → `ab` (quotes stripped correctly)

This breaks data access patterns, as users must include quotes when accessing fields:
```python
data3["myattr"]  # Fails!
data3['"myattr"']  # Required (awkward)
```

## Fix

The issue is in the attribute name parsing logic. Both single and double quotes should be consistently stripped. The bug appears to be in `scipy/io/arff/_arffread.py` in the attribute parsing code.

A proper fix would ensure that:
- Both single and double quotes are stripped consistently
- Quote stripping works regardless of the attribute name length
- The stripped name is used for both metadata and the data array field names