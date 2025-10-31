# Bug Report: flask.json.dumps() Sort Keys Inconsistency

**Target**: `flask.json.dumps`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`flask.json.dumps()` produces different JSON string output for the same dictionary input depending on whether a Flask app context is active, violating the principle of least surprise and causing inconsistent serialization behavior.

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

# Run the test
if __name__ == "__main__":
    test_dumps_consistent_with_without_context()
```

<details>

<summary>
**Failing input**: `{'0': 0, '': 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 21, in <module>
    test_dumps_consistent_with_without_context()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 8, in test_dumps_consistent_with_without_context
    def test_dumps_consistent_with_without_context(d):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 14, in test_dumps_consistent_with_without_context
    assert without_context == with_context, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: flask.json.dumps() behavior should not depend on app context. Got different results: {"0": 0, "": 0} vs {"": 0, "0": 0}
Falsifying example: test_dumps_consistent_with_without_context(
    d={'0': 0, '': 0},
)
```
</details>

## Reproducing the Bug

```python
from flask import Flask
import flask.json

app = Flask(__name__)

test_dict = {'z': 1, 'a': 2, 'b': 3}

print("Testing flask.json.dumps() behavior with and without app context:")
print("=" * 60)
print(f"Input dictionary: {test_dict}")
print()

# Without app context
without_context = flask.json.dumps(test_dict)
print(f"Without app context: {without_context}")

# With app context
with app.app_context():
    with_context = flask.json.dumps(test_dict)
    print(f"With app context:    {with_context}")

print()
print("Comparison:")
print(f"  Outputs are {'SAME' if without_context == with_context else 'DIFFERENT'}")
if without_context != with_context:
    print("  Without context maintains insertion order: 'z' comes before 'a'")
    print("  With context sorts keys alphabetically: 'a' comes before 'z'")
```

<details>

<summary>
Different JSON strings for same dictionary input
</summary>
```
Testing flask.json.dumps() behavior with and without app context:
============================================================
Input dictionary: {'z': 1, 'a': 2, 'b': 3}

Without app context: {"z": 1, "a": 2, "b": 3}
With app context:    {"a": 2, "b": 3, "z": 1}

Comparison:
  Outputs are DIFFERENT
  Without context maintains insertion order: 'z' comes before 'a'
  With context sorts keys alphabetically: 'a' comes before 'z'
```
</details>

## Why This Is A Bug

The function `flask.json.dumps()` exhibits inconsistent behavior based on whether an app context is active, which violates user expectations and the principle of least surprise. Specifically:

1. **Different Code Paths with Different Defaults**:
   - With app context: Uses `DefaultJSONProvider.dumps()` which sets `sort_keys=True` by default (flask/json/provider.py:178)
   - Without app context: Uses standard `json.dumps()` with only the `default` parameter set, no `sort_keys` (flask/json/__init__.py:43-44)

2. **Documentation Gap**: While the documentation mentions that different implementations are used (`current_app.json.dumps()` vs `json.dumps()`), it does NOT warn users that this leads to different serialization behavior (sorted vs unsorted keys).

3. **Real-World Impact**: This inconsistency affects:
   - **Testing**: Unit tests that serialize dicts outside of request contexts may pass but fail in production with app context
   - **Caching**: Systems that generate cache keys from JSON serialization will produce different keys for the same data
   - **API Consistency**: The same API endpoint could return differently formatted JSON based on execution context
   - **Debugging**: Developers may see different output when debugging vs production

4. **Violation of Function Contract**: A pure function like `dumps()` should produce consistent output for the same input, regardless of implicit context.

## Relevant Context

The issue stems from the `DefaultJSONProvider` class having `sort_keys = True` as a class attribute (line 150 in flask/json/provider.py):

```python
class DefaultJSONProvider(JSONProvider):
    # ...
    sort_keys = True
    """Sort the keys in any serialized dicts. This may be useful for
    some caching situations, but can be disabled for better performance.
    When enabled, keys must all be strings, they are not converted
    before sorting.
    """
```

However, when no app context is active, the fallback code in flask/json/__init__.py only sets the `default` parameter:

```python
def dumps(obj: t.Any, **kwargs: t.Any) -> str:
    if current_app:
        return current_app.json.dumps(obj, **kwargs)

    kwargs.setdefault("default", _default)  # Only sets default, not sort_keys
    return _json.dumps(obj, **kwargs)
```

## Proposed Fix

Make the behavior consistent by ensuring `sort_keys=True` is set even when there's no app context, matching the DefaultJSONProvider behavior:

```diff
--- a/flask/json/__init__.py
+++ b/flask/json/__init__.py
@@ -41,6 +41,7 @@ def dumps(obj: t.Any, **kwargs: t.Any) -> str:
         return current_app.json.dumps(obj, **kwargs)

     kwargs.setdefault("default", _default)
+    kwargs.setdefault("sort_keys", True)
     return _json.dumps(obj, **kwargs)
```