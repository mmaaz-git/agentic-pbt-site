# Bug Report: scipy.io.arff NumericAttribute Question Mark Parsing

**Target**: `scipy.io.arff._arffread.NumericAttribute.parse_data`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `NumericAttribute.parse_data` method incorrectly treats any string containing a `?` character as missing data (NaN), instead of only treating standalone `?` as missing data. This causes malformed numeric values like `"1.5?"` or `"?1.5"` to be silently parsed as NaN instead of raising an appropriate error.

## Property-Based Test

```python
from io import StringIO

import numpy as np
from hypothesis import assume, given, settings, strategies as st
from scipy.io import arff


@given(
    valid_float=st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10
    ),
    question_location=st.sampled_from(['prefix', 'suffix', 'inside'])
)
@settings(max_examples=300)
def test_question_mark_only_valid_for_standalone(valid_float, question_location):
    valid_str = str(valid_float)

    if question_location == 'prefix':
        invalid_value = '?' + valid_str
    elif question_location == 'suffix':
        invalid_value = valid_str + '?'
    else:
        if len(valid_str) < 2:
            assume(False)
        mid = len(valid_str) // 2
        invalid_value = valid_str[:mid] + '?' + valid_str[mid:]

    assume(invalid_value.strip() != '?')

    content = f"""@relation test
@attribute val numeric
@data
{invalid_value}
"""

    f = StringIO(content)
    data, meta = arff.loadarff(f)

    result = data['val'][0]
    if np.isnan(result):
        assert False, f"Value {invalid_value!r} incorrectly parsed as NaN; should raise error"
```

**Failing input**: `'?0.0'`

## Reproducing the Bug

```python
from io import StringIO

import numpy as np
from scipy.io import arff

content = """@relation test
@attribute val numeric
@data
?0.0
"""

f = StringIO(content)
data, meta = arff.loadarff(f)

result = data['val'][0]
print(f"Input: '?0.0'")
print(f"Result: {result}")
print(f"Is NaN: {np.isnan(result)}")
```

Output:
```
Input: '?0.0'
Result: nan
Is NaN: True
```

## Why This Is A Bug

According to the ARFF specification, missing data is represented by a standalone `?` character. The current implementation uses `if '?' in data_str:` which matches ANY string containing `?`, not just the missing data marker `?`.

This causes invalid ARFF data like `"?0.0"`, `"1.5?"`, or `"1.?5"` to be silently accepted as missing data (NaN) instead of raising a `ValueError`. This could mask data corruption issues where malformed numeric values are present in ARFF files.

The docstring for `NumericAttribute.parse_data` shows:
- `parse_data('?')` → `nan` ✓ correct
- `parse_data('?\n')` → `nan` ✓ correct (handles trailing whitespace)

But the current implementation would also do:
- `parse_data('?0.0')` → `nan` ✗ should raise ValueError
- `parse_data('1.5?')` → `nan` ✗ should raise ValueError

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -196,7 +196,7 @@ class NumericAttribute(Attribute):
     def parse_data(self, data_str):
         """
         Parse a value of this type.

         Parameters
         ----------
         data_str : str
            string to convert

         Returns
         -------
         f : float
            where float can be nan

         Examples
         --------
         >>> from scipy.io.arff._arffread import NumericAttribute
         >>> atr = NumericAttribute('atr')
         >>> atr.parse_data('1')
         1.0
         >>> atr.parse_data('1\\n')
         1.0
         >>> atr.parse_data('?\\n')
         nan
         """
-        if '?' in data_str:
+        if data_str.strip() == '?':
             return np.nan
         else:
             return float(data_str)
```