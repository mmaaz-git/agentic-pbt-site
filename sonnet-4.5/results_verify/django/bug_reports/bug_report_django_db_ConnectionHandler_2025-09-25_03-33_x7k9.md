# Bug Report: django.db.utils.ConnectionHandler Type Assumption on TEST Setting

**Target**: `django.db.utils.ConnectionHandler.configure_settings`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ConnectionHandler.configure_settings` method assumes the `TEST` database configuration value is always a dictionary, causing an `AttributeError` when users provide non-dictionary values like empty strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.db.utils import ConnectionHandler
import copy

@given(st.one_of(
    st.just({}),
    st.dictionaries(
        st.just('default'),
        st.dictionaries(
            st.sampled_from(['ENGINE', 'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT', 'OPTIONS', 'TEST']),
            st.one_of(st.text(), st.dictionaries(st.text(), st.text()), st.booleans(), st.integers()),
            max_size=5
        ),
        min_size=1,
        max_size=1
    ).flatmap(lambda default_db: st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(lambda x: x != 'default'),
        st.dictionaries(
            st.sampled_from(['ENGINE', 'NAME', 'USER']),
            st.text(max_size=50),
            max_size=3
        ),
        max_size=5
    ).map(lambda other_dbs: {**default_db, **other_dbs}))
))
@settings(max_examples=500)
def test_configure_settings_idempotence(databases):
    handler = ConnectionHandler()
    configured_once = handler.configure_settings(copy.deepcopy(databases))
    configured_twice = handler.configure_settings(copy.deepcopy(configured_once))
    assert configured_once == configured_twice
```

**Failing input**: `databases={'default': {'TEST': ''}}`

## Reproducing the Bug

```python
from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
configured = handler.configure_settings(databases)
```

**Output**:
```
AttributeError: 'str' object has no attribute 'setdefault'
```

## Why This Is A Bug

The code at `/django/db/utils.py:172-181` calls `conn.setdefault("TEST", {})` which returns the existing value if `TEST` is already set. When `TEST` is a non-dict value (like an empty string), the subsequent call to `test_settings.setdefault(key, value)` fails because strings don't have a `setdefault` method.

This violates user expectations because:
1. Users receive a cryptic `AttributeError` instead of a clear configuration error
2. The code doesn't validate that `TEST` must be a dictionary type
3. Other configuration keys accept various types, making this inconsistent

## Fix

```diff
--- a/django/db/utils.py
+++ b/django/db/utils.py
@@ -169,7 +169,10 @@ class ConnectionHandler(BaseConnectionHandler):
             for setting in ["NAME", "USER", "PASSWORD", "HOST", "PORT"]:
                 conn.setdefault(setting, "")

-            test_settings = conn.setdefault("TEST", {})
+            test_settings = conn.get("TEST", {})
+            if not isinstance(test_settings, dict):
+                test_settings = {}
+            conn["TEST"] = test_settings
             default_test_settings = [
                 ("CHARSET", None),
                 ("COLLATION", None),
```