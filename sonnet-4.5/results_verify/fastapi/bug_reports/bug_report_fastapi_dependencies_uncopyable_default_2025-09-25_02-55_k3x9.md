# Bug Report: fastapi.dependencies Uncopyable Default Crash

**Target**: `fastapi.dependencies.utils._get_multidict_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

FastAPI crashes with `TypeError` when a query parameter with an uncopyable default value is not provided in the request. The bug occurs in `_get_multidict_value` which attempts to deepcopy the default value, failing for objects that don't support deepcopy.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient


class UncopyableClass:
    def __init__(self, value: int = 42):
        self.value = value

    def __deepcopy__(self, memo):
        raise TypeError("Cannot deepcopy this object")


@given(st.integers())
@settings(max_examples=10)
def test_uncopyable_defaults_should_not_crash(value):
    app = FastAPI()
    uncopyable = UncopyableClass(value)

    @app.get("/test")
    def endpoint(param: int = Query(default=uncopyable)):
        return {"param": param}

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
```

**Failing input**: Any uncopyable object used as a default value

## Reproducing the Bug

```python
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient


class UncopyableDefault:
    def __init__(self, value: int = 42):
        self.value = value

    def __deepcopy__(self, memo):
        raise TypeError("Cannot deepcopy this object")


app = FastAPI()
uncopyable = UncopyableDefault()


@app.get("/test")
def endpoint(param: int = Query(default=uncopyable)):
    return {"param": param}


client = TestClient(app)
response = client.get("/test")
```

Running this code produces:

```
TypeError: Cannot deepcopy this object
  File "/fastapi/dependencies/utils.py", line 736, in _get_multidict_value
    return deepcopy(field.default)
```

## Why This Is A Bug

The crash occurs at runtime when handling a valid API request. While uncommon, uncopyable objects (e.g., singletons, locks, database connections) can legitimately be used as defaults. The deepcopy is unnecessary in this context - defaults can be returned directly since they're already validated by Pydantic during setup.

The issue manifests only when:
1. A parameter has an uncopyable default value
2. The parameter is not provided in the request

## Fix

The issue is in `fastapi/dependencies/utils.py` at line 736 and also line 705 in `_validate_value_with_model_field`. The deepcopy calls should either be removed (defaults are already properly handled by Pydantic) or wrapped in a try-except to fall back to direct return.

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -702,7 +702,10 @@ def _validate_value_with_model_field(
         if field.required:
             return None, [get_missing_field_error(loc=loc)]
         else:
-            return deepcopy(field.default), []
+            try:
+                return deepcopy(field.default), []
+            except (TypeError, AttributeError):
+                return field.default, []
     v_, errors_ = field.validate(value, values, loc=loc)
     if isinstance(errors_, ErrorWrapper):
         return None, [errors_]
@@ -733,7 +736,10 @@ def _get_multidict_value(
         if field.required:
             return
         else:
-            return deepcopy(field.default)
+            try:
+                return deepcopy(field.default)
+            except (TypeError, AttributeError):
+                return field.default
     return value
```