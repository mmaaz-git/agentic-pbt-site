# Bug Report: Django DictionarySerializer TypeError with Mixed-Type Keys

**Target**: `django.db.migrations.serializer.DictionarySerializer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

DictionarySerializer crashes with TypeError when attempting to serialize dictionaries with keys of incomparable types (e.g., integers and strings mixed together) due to calling `sorted()` without a key function in Python 3.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from django.db.migrations.serializer import serializer_factory

@given(st.dictionaries(
    st.one_of(st.integers(), st.text()),
    st.integers(),
    min_size=2,
    max_size=5
))
@example({1: 10, 'a': 20})
def test_dict_serializer_mixed_keys(value):
    serialized, imports = serializer_factory(value).serialize()
    exec_globals = {}
    for imp in imports:
        exec(imp, exec_globals)
    deserialized = eval(serialized, exec_globals)
    assert deserialized == value
```

**Failing input**: `{1: 'value1', 'key2': 'value2'}`

## Reproducing the Bug

```python
from django.db.migrations.serializer import serializer_factory

test_dict = {1: 'value1', 'key2': 'value2'}
serialized, imports = serializer_factory(test_dict).serialize()
```

**Output**:
```
TypeError: '<' not supported between instances of 'str' and 'int'
```

## Why This Is A Bug

In Python 3, you cannot compare values of incompatible types like `int` and `str` using comparison operators. The `DictionarySerializer.serialize()` method at line 129 calls `sorted(self.value.items())` without a key function. When the dictionary contains keys of mixed types (e.g., both integers and strings), the `sorted()` function attempts to compare these keys directly, resulting in a TypeError.

While dictionaries with mixed-type keys are uncommon in Django migrations, they are valid Python dictionaries and should be serializable. This bug could occur if:
- A model field has a default value that is a dict with mixed-type keys
- Custom migration operations use such dictionaries
- Third-party Django apps use mixed-type keys in field configurations

## Fix

```diff
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -126,7 +126,7 @@ class DictionarySerializer(BaseSerializer):
     def serialize(self):
         imports = set()
         strings = []
-        for k, v in sorted(self.value.items()):
+        for k, v in sorted(self.value.items(), key=lambda item: repr(item[0])):
             k_string, k_imports = serializer_factory(k).serialize()
             v_string, v_imports = serializer_factory(v).serialize()
             imports.update(k_imports)
```

This fix sorts dictionary items by the string representation of the key using `repr()`, which works for all types and ensures deterministic serialization order. This approach is consistent with `BaseUnorderedSequenceSerializer` (line 60), which also uses `key=repr` for sorting.