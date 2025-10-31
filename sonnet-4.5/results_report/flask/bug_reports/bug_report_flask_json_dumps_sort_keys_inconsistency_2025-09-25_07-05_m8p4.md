# Bug Report: Flask json.dumps() Sort Keys Inconsistency

**Target**: `flask.json.dumps`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`flask.json.dumps()` produces different output for the same dict depending on whether an app context is active. With an app context, keys are sorted alphabetically. Without an app context, keys maintain insertion order. This behavioral inconsistency violates the principle of least surprise.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask import Flask
import flask.json

app = Flask(__name__)

@given(st.dictionaries(st.text(), st.integers(), min_size=2))
def test_dumps_consistent_with_without_context(d):
    without_context = flask.json.dumps(d)

    with app.app_context():
        with_context = flask.json.dumps(d)

    assert without_context == with_context, (
        f"flask.json.dumps() behavior should not depend on app context. "
        f"Got different results: {without_context} vs {with_context}"
    )
```

**Failing input**: Any dict with multiple keys, e.g., `{'z': 1, 'a': 2}`

## Reproducing the Bug

```python
from flask import Flask
import flask.json

app = Flask(__name__)

test_dict = {'z': 1, 'a': 2, 'b': 3}

without_context = flask.json.dumps(test_dict)
print(f"Without context: {without_context}")

with app.app_context():
    with_context = flask.json.dumps(test_dict)
    print(f"With context:    {with_context}")
```

Output:
```
Without context: {"z": 1, "a": 2, "b": 3}
With context:    {"a": 2, "b": 3, "z": 1}
```

## Why This Is A Bug

The function's behavior should be consistent regardless of whether an app context is active. Currently:

**With app context:** Uses `DefaultJSONProvider.dumps()` which sets `sort_keys=True` by default (flask/json/provider.py:178)

**Without app context:** Uses standard `json.dumps()` with only the `default` parameter set, no `sort_keys` (flask/json/__init__.py:43-44)

This creates unpredictable behavior where the same code produces different output depending on execution context. The documentation for `flask.json.dumps()` makes no mention of this context-dependent behavior.

This particularly affects:
1. Testing code that serializes dicts outside of request contexts
2. Utility functions that call `flask.json.dumps()` in different contexts
3. Caching systems that rely on consistent serialization

## Fix

Make the behavior consistent by setting `sort_keys=True` even when there's no app context:

```diff
def dumps(obj: t.Any, **kwargs: t.Any) -> str:
    if current_app:
        return current_app.json.dumps(obj, **kwargs)

    kwargs.setdefault("default", _default)
+   kwargs.setdefault("sort_keys", True)
    return _json.dumps(obj, **kwargs)
```

This ensures `flask.json.dumps()` always produces the same output regardless of context, matching the behavior of `DefaultJSONProvider`.