# Bug Report: scipy.io.arff NominalAttribute Unicode Support

**Target**: `scipy.io.arff._arffread.NominalAttribute`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `NominalAttribute` class uses `np.bytes_` for storing nominal attribute values, which causes a `UnicodeEncodeError` crash when ARFF files contain non-ASCII Unicode characters in nominal attributes. This affects real-world datasets with international names, places, or categories.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.io import arff
from io import StringIO


@given(st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=('L',), blacklist_characters=',{}'),
        min_size=1,
        max_size=10
    ),
    min_size=1,
    max_size=5,
    unique=True
))
def test_nominal_unicode_support(values):
    """
    Property: ARFF files with nominal attributes should support Unicode values.
    """
    assume(all(v for v in values))

    arff_content = f"""@relation test
@attribute category {{{",".join(values)}}}
@data
{values[0]}
"""

    data, meta = arff.loadarff(StringIO(arff_content))
    assert len(data) == 1
```

**Failing input**: `values=['ª']` (or any string with non-ASCII characters like 'café', 'naïve', etc.)

## Reproducing the Bug

```python
from scipy.io import arff
from io import StringIO

arff_content = """@relation test
@attribute category {café,naïve}
@data
café
"""

data, meta = arff.loadarff(StringIO(arff_content))
```

Output:
```
UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 3: ordinal not in range(128)
```

## Why This Is A Bug

The code uses `np.bytes_` which stores data as ASCII-encoded byte strings. This causes crashes when processing valid Unicode text that is common in real-world datasets (e.g., names like "José", cities like "São Paulo", or categories like "café").

The problematic code at line 103 of `_arffread.py`:

```python
class NominalAttribute(Attribute):
    def __init__(self, name, values):
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.bytes_, max(len(i) for i in values))  # BUG: ASCII-only
```

This violates user expectations because:
1. The documentation doesn't mention ASCII-only limitation
2. Unicode works fine in attribute names (only values are affected)
3. Modern ARFF files can be UTF-8 encoded
4. The error is a cryptic UnicodeEncodeError from NumPy, not a clear message

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -100,7 +100,7 @@ class NominalAttribute(Attribute):
         super().__init__(name)
         self.values = values
         self.range = values
-        self.dtype = (np.bytes_, max(len(i) for i in values))
+        self.dtype = (np.str_, max(len(i) for i in values))

     @staticmethod
     def _get_nom_val(atrv):
```

This changes the dtype from `np.bytes_` (ASCII byte strings) to `np.str_` (Unicode strings), which properly handles all Unicode characters.