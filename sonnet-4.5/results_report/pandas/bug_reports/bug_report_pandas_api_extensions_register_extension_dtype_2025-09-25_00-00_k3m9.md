# Bug Report: pandas.api.extensions.register_extension_dtype Non-Idempotent Registration

**Target**: `pandas.api.extensions.register_extension_dtype` and `pandas.core.dtypes.base.Registry.register`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `register_extension_dtype()` multiple times on the same dtype class adds duplicate entries to the global dtype registry, causing memory waste and violating typical registration system expectations of idempotency.

## Property-Based Test

```python
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.base import _registry
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=10))
def test_registration_idempotency(n_registrations):
    class TestDtype(ExtensionDtype):
        name = "test_dtype_idempotent"

        @property
        def type(self):
            return object

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import ExtensionArray
            return ExtensionArray

    initial_size = len(_registry.dtypes)

    for _ in range(n_registrations):
        register_extension_dtype(TestDtype)

    final_size = len(_registry.dtypes)

    assert final_size - initial_size == 1
```

**Failing input**: Any `n_registrations > 1`

## Reproducing the Bug

```python
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.base import _registry


class MyDtype(ExtensionDtype):
    name = "mydtype"

    @property
    def type(self):
        return object

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import ExtensionArray
        return ExtensionArray


initial_size = len(_registry.dtypes)
print(f"Initial registry size: {initial_size}")

register_extension_dtype(MyDtype)
print(f"After first registration: {len(_registry.dtypes)}")

register_extension_dtype(MyDtype)
print(f"After second registration: {len(_registry.dtypes)}")

print(f"\nDuplicate entries: {len(_registry.dtypes) - initial_size - 1}")
```

## Why This Is A Bug

The `Registry.register()` method blindly appends dtypes without checking for duplicates. This violates the expectation that registration operations are idempotent. Realistic scenarios where this occurs:

1. **Module reloading during development**: When developers reload modules containing `@register_extension_dtype` decorators
2. **Accidental double decoration**: If the decorator is applied twice by mistake
3. **Programmatic registration**: Code that calls `register_extension_dtype()` multiple times

While this doesn't break functionality (Registry.find() returns on first match), it:
- Wastes memory by storing duplicate references
- Degrades performance slightly as the registry grows unnecessarily
- Violates principle of least surprise (most registration systems are idempotent)

## Fix

```diff
--- a/pandas/core/dtypes/base.py
+++ b/pandas/core/dtypes/base.py
@@ -520,7 +520,10 @@ class Registry:
     def register(self, dtype: type_t[ExtensionDtype]) -> None:
         """
         Parameters
         ----------
         dtype : ExtensionDtype class
         """
         if not issubclass(dtype, ExtensionDtype):
             raise ValueError("can only register pandas extension dtypes")

+        if dtype not in self.dtypes:
             self.dtypes.append(dtype)
```