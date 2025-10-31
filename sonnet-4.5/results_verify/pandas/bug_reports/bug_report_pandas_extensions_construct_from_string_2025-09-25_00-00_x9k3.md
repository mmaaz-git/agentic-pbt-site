# Bug Report: ExtensionDtype.construct_from_string AssertionError with property-based name

**Target**: `pandas.api.extensions.ExtensionDtype.construct_from_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

ExtensionDtype.construct_from_string() fails with an AssertionError when a subclass implements `name` as a property instead of a class attribute, despite the base class defining `name` as an abstract property.

## Property-Based Test

```python
from pandas.api.extensions import ExtensionDtype


class PropertyNameDtype(ExtensionDtype):
    type = str
    _test_name = "property_name_dtype"

    @property
    def name(self):
        return self._test_name

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import ExtensionArray
        return ExtensionArray


def test_construct_from_string_with_property_name():
    dtype = PropertyNameDtype()

    with pytest.raises(AssertionError) as exc_info:
        result = dtype.construct_from_string("property_name_dtype")

    assert "property" in str(exc_info.value).lower() or "str" in str(exc_info.value)
```

**Failing input**: `ExtensionDtype subclass with name implemented as a property`

## Reproducing the Bug

```python
from pandas.api.extensions import ExtensionDtype


class PropertyNameDtype(ExtensionDtype):
    type = str

    @property
    def name(self):
        return "my_custom_dtype"

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import ExtensionArray
        return ExtensionArray


dtype = PropertyNameDtype()
print(f"Created dtype with name: {dtype.name}")

try:
    result = dtype.construct_from_string("my_custom_dtype")
    print(f"Success: {result}")
except AssertionError as e:
    print(f"AssertionError: {e}")
    print(f"Expected: name to work as property, got assertion failure")
```

## Why This Is A Bug

The base `ExtensionDtype` class defines `name` as an abstract property (line 191-198 in pandas/core/dtypes/base.py), which naturally leads developers to implement it as a property in their subclasses. However, `construct_from_string` contains an assertion at line 289 that assumes `cls.name` is a string (class attribute), not a property object:

```python
assert isinstance(cls.name, str), (cls, type(cls.name))
```

When `name` is implemented as a property, `cls.name` returns a property object, not a string, causing the assertion to fail. This violates the principle of least surprise and creates an undocumented restriction that contradicts the base class design.

## Fix

Change the assertion to handle both class attributes and properties by accessing the name through an instance:

```diff
diff --git a/pandas/core/dtypes/base.py b/pandas/core/dtypes/base.py
index 1234567..abcdefg 100644
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -286,7 +286,10 @@ class ExtensionDtype:
             raise TypeError(
                 f"'construct_from_string' expects a string, got {type(string)}"
             )
-        assert isinstance(cls.name, str), (cls, type(cls.name))
+        # Handle both class attributes and properties for name
+        name_value = cls.name if isinstance(cls.name, str) else cls().name
+        assert isinstance(name_value, str), (cls, type(name_value))
+
-        if string != cls.name:
+        if string != name_value:
             raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
         return cls()
```

Alternatively, document that `name` must be a class attribute, not a property, to avoid confusion.