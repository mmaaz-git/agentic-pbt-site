# Bug Report: flask.config.import_string Silent Mode Fails

**Target**: `flask.config.import_string`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `import_string` function violates its documented contract when `silent=True` by raising a `ValueError` instead of returning `None` for certain invalid inputs. Specifically, when the input is a single colon (`':'`), the function raises `ValueError: Empty module name` even when `silent=True`.

## Property-Based Test

```python
from flask.config import import_string
from hypothesis import given, strategies as st, settings


@given(num_colons=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_import_string_silent_mode_catches_all_errors(num_colons):
    """
    Property: When silent=True, import_string should return None for any
    import failure, not raise exceptions.
    """
    invalid_path = ':' * num_colons

    result = import_string(invalid_path, silent=True)

    assert result is None
```

**Failing input**: `':'` (single colon)

## Reproducing the Bug

```python
from flask.config import import_string

result = import_string(':', silent=True)
```

Expected: Returns `None`
Actual: Raises `ValueError: Empty module name`

## Why This Is A Bug

The function's docstring states:

> If `silent` is True the return value will be `None` if the import fails.

And:

> :param silent: if set to `True` import errors are ignored and `None` is returned instead.

When a user passes `silent=True`, they expect the function to handle all import failures gracefully by returning `None`. However, the function only catches `ImportError`, not other exceptions that can occur during the import process like `ValueError`.

The root cause:
1. Input `':'` is converted to `'.'` by `import_name.replace(":", ".")`
2. `__import__('.')` raises `ValueError('Empty module name')`
3. This `ValueError` is not caught by the `except ImportError` block
4. The exception propagates to the caller, violating the API contract

## Fix

```diff
diff --git a/src/flask/config.py b/src/flask/config.py
index 1234567..abcdefg 100644
--- a/src/flask/config.py
+++ b/src/flask/config.py
@@ -150,7 +150,7 @@ def import_string(import_name: str, silent: bool = False) -> t.Any:
         except AttributeError as e:
             raise ImportError(e) from None

-    except ImportError as e:
+    except (ImportError, ValueError) as e:
         if not silent:
             raise ImportStringError(import_name, e).with_traceback(
                 sys.exc_info()[2]
```

The fix catches `ValueError` in addition to `ImportError`, ensuring that all import failures are handled consistently when `silent=True`.