# Bug Report: django.conf.SettingsReference String Operations Lose setting_name

**Target**: `django.conf.SettingsReference`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`SettingsReference` is a `str` subclass with a `setting_name` attribute used for serializing model references in Django migrations. However, any string operation (`.upper()`, `.lower()`, `.strip()`, etc.) returns a plain `str` object that loses the `setting_name` attribute, breaking serialization and potentially causing migration failures.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.conf import SettingsReference


@settings(max_examples=1000)
@given(st.text(), st.text(min_size=1).filter(lambda x: x.isupper() and x.isidentifier()))
def test_settings_reference_preserves_setting_name_after_string_ops(value, setting_name):
    ref = SettingsReference(value, setting_name)

    string_ops = [
        (ref.upper, lambda: ref.upper()),
        (ref.lower, lambda: ref.lower()),
        (ref.strip, lambda: ref.strip()),
        (ref.replace, lambda: ref.replace('a', 'b') if 'a' in ref else ref),
        (ref.capitalize, lambda: ref.capitalize()),
    ]

    for op_name, op in string_ops:
        result = op()
        assert hasattr(result, 'setting_name'), (
            f"{op_name.__name__}() lost setting_name attribute"
        )
        assert result.setting_name == setting_name
```

**Failing input**: Any SettingsReference object, e.g., `SettingsReference("auth.User", "AUTH_USER_MODEL")`

## Reproducing the Bug

```python
from django.conf import SettingsReference
from django.db.migrations.serializer import SettingsReferenceSerializer

ref = SettingsReference("auth.User", "AUTH_USER_MODEL")
print(f"Original: {ref.setting_name}")

upper_ref = ref.upper()
print(f"After .upper(): {upper_ref.setting_name}")
```

**Output:**
```
Original: AUTH_USER_MODEL
AttributeError: 'str' object has no attribute 'setting_name'
```

**Impact on serialization:**
```python
serializer = SettingsReferenceSerializer(upper_ref)
result, imports = serializer.serialize()
```

**Output:**
```
AttributeError: 'str' object has no attribute 'setting_name'
```

This would break Django's migration system if any code performs string transformations on SettingsReference objects before serialization.

## Why This Is A Bug

1. The docstring for `SettingsReference` states it "serializes to a settings.NAME attribute reference", which requires the `setting_name` attribute to be preserved
2. `SettingsReferenceSerializer.serialize()` accesses `self.value.setting_name` - if this attribute is missing, serialization fails with AttributeError
3. `SettingsReference` is used in `django.db.models.fields.related.py` for swappable models (like AUTH_USER_MODEL) and relies on the `setting_name` attribute being preserved
4. String operations are common and reasonable operations that users might perform, expecting the object to maintain its type and attributes

## Fix

Override string methods to return `SettingsReference` instances instead of plain `str` objects:

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -32,6 +32,30 @@ class SettingsReference(str):
 class SettingsReference(str):
     """
     String subclass which references a current settings value. It's treated as
     the value in memory but serializes to a settings.NAME attribute reference.
     """

     def __new__(self, value, setting_name):
         return str.__new__(self, value)

     def __init__(self, value, setting_name):
         self.setting_name = setting_name
+
+    def _make_instance(self, value):
+        return SettingsReference(value, self.setting_name)
+
+    def upper(self):
+        return self._make_instance(str.upper(self))
+
+    def lower(self):
+        return self._make_instance(str.lower(self))
+
+    def capitalize(self):
+        return self._make_instance(str.capitalize(self))
+
+    def title(self):
+        return self._make_instance(str.title(self))
+
+    def swapcase(self):
+        return self._make_instance(str.swapcase(self))
+
+    def strip(self, chars=None):
+        return self._make_instance(str.strip(self, chars))
+
+    def lstrip(self, chars=None):
+        return self._make_instance(str.lstrip(self, chars))
+
+    def rstrip(self, chars=None):
+        return self._make_instance(str.rstrip(self, chars))
+
+    def replace(self, old, new, count=-1):
+        return self._make_instance(str.replace(self, old, new, count))
```

**Note**: A complete fix would need to override all string methods that return new strings (`center`, `ljust`, `rjust`, `zfill`, `format`, `__getitem__` for slicing, etc.). Alternatively, consider using composition instead of inheritance to avoid this class of bugs entirely.