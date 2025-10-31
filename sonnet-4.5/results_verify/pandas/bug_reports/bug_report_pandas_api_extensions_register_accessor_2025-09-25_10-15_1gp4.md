# Bug Report: pandas.api.extensions register_*_accessor Invalid Identifier Acceptance

**Target**: `pandas.api.extensions.register_series_accessor`, `register_dataframe_accessor`, `register_index_accessor`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `register_*_accessor` functions accept invalid Python identifiers as accessor names, including empty strings, names starting with digits, and names containing special characters. This violates the expectation that accessor names should be valid Python identifiers usable with normal attribute syntax.

## Property-Based Test

```python
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pandas.api.extensions import register_series_accessor


@given(name=st.text(min_size=0, max_size=20))
def test_register_accessor_validates_identifier(name):
    class TestAccessor:
        def __init__(self, obj):
            self._obj = obj

    if not name or not name.isidentifier() or name.startswith('_'):
        with pytest.raises(ValueError, match=".*identifier.*"):
            register_series_accessor(name)(TestAccessor)
    else:
        register_series_accessor(name)(TestAccessor)
        s = pd.Series([1, 2, 3])
        assert hasattr(s, name)
```

**Failing input**: `name=''` (and many other invalid identifiers)

## Reproducing the Bug

```python
import pandas as pd
from pandas.api.extensions import register_series_accessor

@register_series_accessor("")
class EmptyNameAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def test(self):
        return "works"

s = pd.Series([1, 2, 3])
accessor = getattr(s, "")
print(accessor.test())

assert "".isidentifier() == False
```

Additional examples of accepted invalid identifiers:
- `"123invalid"` (starts with digit)
- `"with-dash"` (contains hyphen)
- `"with space"` (contains space)
- `"with.dot"` (contains period)

## Why This Is A Bug

1. **Contract Violation**: The function should only accept valid Python identifiers as accessor names, since the intended use is attribute access (e.g., `series.my_accessor`).

2. **Unusable API**: Accessors registered with invalid identifiers cannot be used with normal Python syntax. For example, `s.` is a syntax error, and `s.123invalid` is invalid syntax.

3. **Inconsistent Validation**: While the function warns about overriding existing attributes, it fails to validate the most basic requirement - that the name is a valid identifier.

4. **Documentation Implies Valid Identifiers**: The examples in the docstring all use valid Python identifiers, implying this is the expected input domain.

## Fix

```diff
--- a/pandas/core/accessor.py
+++ b/pandas/core/accessor.py
@@ -180,6 +180,10 @@ class CachedAccessor:
 def _register_accessor(name, cls):
     def decorator(accessor):
+        if not isinstance(name, str) or not name.isidentifier():
+            raise ValueError(
+                f"Accessor name must be a valid Python identifier, got {name!r}"
+            )
         if hasattr(cls, name):
             warnings.warn(
                 f"registration of accessor {accessor!r} under name "
```

Note: The exact line numbers may vary depending on the pandas version, but the fix should be added at the beginning of the `_register_accessor` function before any attribute setting occurs.