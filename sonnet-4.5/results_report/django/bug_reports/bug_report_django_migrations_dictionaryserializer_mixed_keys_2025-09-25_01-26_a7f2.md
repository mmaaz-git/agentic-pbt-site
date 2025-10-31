# Bug Report: django.db.migrations DictionarySerializer Mixed Key Types

**Target**: `django.db.migrations.serializer.DictionarySerializer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`DictionarySerializer.serialize()` crashes with a `TypeError` when attempting to serialize dictionaries with mixed-type keys (e.g., integers and strings) because it calls `sorted()` without a key function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.migrations.serializer import DictionarySerializer

@given(st.dictionaries(
    st.one_of(st.integers(), st.text()),
    st.one_of(st.integers(), st.text()),
    max_size=10
))
def test_dictionary_serializer_deterministic(d):
    serializer = DictionarySerializer(d)
    result, _ = serializer.serialize()
```

**Failing input**: `{0: 0, '': 0}`

## Reproducing the Bug

```python
from django.db.migrations.serializer import DictionarySerializer

d = {0: 0, '': 0}
serializer = DictionarySerializer(d)
result, imports = serializer.serialize()
```

**Output:**
```
TypeError: '<' not supported between instances of 'str' and 'int'
```

## Why This Is A Bug

The `DictionarySerializer` attempts to create deterministic output by sorting dictionary items (line 129 of serializer.py). However, in Python 3, comparing values of incompatible types (like `int` and `str`) raises a `TypeError`. This means any dictionary with heterogeneous key types will crash the serializer.

While mixed-type dictionary keys may be uncommon in Django migrations, they are valid Python dictionaries and the serializer should handle them gracefully. A user could legitimately pass such a dictionary as a field option or model parameter.

## Fix

```diff
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -126,7 +126,7 @@ class DictionarySerializer(BaseSerializer):
     def serialize(self):
         imports = set()
         strings = []
-        for k, v in sorted(self.value.items()):
+        for k, v in sorted(self.value.items(), key=lambda x: (type(x[0]).__name__, x[0])):
             k_string, k_imports = serializer_factory(k).serialize()
             v_string, v_imports = serializer_factory(v).serialize()
             imports.update(k_imports)
```

This fix sorts items first by type name (ensuring same-type items are grouped together), then by value within each type group. This provides deterministic ordering even for mixed-type dictionaries.