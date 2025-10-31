# Bug Report: Deserializer Raises KeyError Instead of DeserializationError

**Target**: `django.core.serializers.python.Deserializer`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The Python/JSON deserializer raises KeyError instead of DeserializationError when deserializing malformed JSON that is missing required keys like "model" or "fields".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core import serializers
from django.core.serializers.base import DeserializationError
import json

@given(st.dictionaries(st.text(), st.text()).filter(lambda d: 'model' not in d or 'fields' not in d))
def test_deserializer_handles_malformed_json(malformed_dict):
    json_str = json.dumps([malformed_dict])
    try:
        list(serializers.deserialize('json', json_str))
    except Exception as e:
        assert isinstance(e, DeserializationError), f"Should raise DeserializationError, not {type(e).__name__}"
```

**Failing input**: `[{"fields": {}}]` (missing "model" key)

## Reproducing the Bug

```python
from django.core import serializers
from django.core.serializers.base import DeserializationError

try:
    list(serializers.deserialize('json', '[{"fields": {}}]'))
except KeyError as e:
    print(f"KeyError raised: {e}")
except DeserializationError as e:
    print(f"DeserializationError raised: {e}")
```

**Output:**
```
KeyError raised: 'model'
```

## Why This Is A Bug

The deserializer's `_handle_object` method (python.py:130-202) accesses dictionary keys without checking if they exist:

- Line 137: `Model = self._get_model_from_node(obj["model"])` - raises KeyError if "model" key missing
- Line 155: `for field_name, field_value in obj["fields"].items():` - raises KeyError if "fields" key missing

The API contract states that deserialization errors should raise `DeserializationError`, not raw `KeyError`. Users expect to catch `DeserializationError` to handle invalid input, but `KeyError` bypasses this expectation and can crash applications.

## Fix

```diff
--- a/django/core/serializers/python.py
+++ b/django/core/serializers/python.py
@@ -134,7 +134,13 @@ class Deserializer(base.Deserializer):

         # Look up the model and starting build a dict of data for it.
         try:
-            Model = self._get_model_from_node(obj["model"])
+            try:
+                model_identifier = obj["model"]
+            except (KeyError, TypeError):
+                raise base.DeserializationError(
+                    f"Missing or invalid 'model' key in deserialized object: {obj}"
+                )
+            Model = self._get_model_from_node(model_identifier)
         except base.DeserializationError:
             if self.ignorenonexistent:
                 return
@@ -152,7 +158,13 @@ class Deserializer(base.Deserializer):
         field_names = self.field_names_cache[Model]

         # Handle each field
-        for field_name, field_value in obj["fields"].items():
+        try:
+            fields = obj["fields"]
+        except (KeyError, TypeError):
+            raise base.DeserializationError(
+                f"Missing or invalid 'fields' key in deserialized object: {obj}"
+            )
+        for field_name, field_value in fields.items():
             if self.ignorenonexistent and field_name not in field_names:
                 # skip fields no longer on model
                 continue
```