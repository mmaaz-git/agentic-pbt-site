# Bug Report: django.db.utils.ConnectionHandler.configure_settings TEST Setting Type Validation

**Target**: `django.db.utils.ConnectionHandler.configure_settings`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `configure_settings` method in `ConnectionHandler` crashes with an `AttributeError` when the `TEST` setting is provided as a non-dict value (e.g., empty string, integer, or any non-dict type), because it assumes `TEST` is always a dictionary without validation.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.db.utils import ConnectionHandler, DEFAULT_DB_ALIAS


def make_database_config_strategy():
    db_config = st.dictionaries(
        st.sampled_from(['ENGINE', 'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT',
                         'ATOMIC_REQUESTS', 'AUTOCOMMIT', 'CONN_MAX_AGE',
                         'CONN_HEALTH_CHECKS', 'OPTIONS', 'TIME_ZONE', 'TEST']),
        st.one_of(
            st.text(max_size=100),
            st.booleans(),
            st.integers(min_value=0, max_value=1000),
            st.dictionaries(st.text(max_size=20), st.text(max_size=100), max_size=5),
            st.none()
        ),
        max_size=10
    )

    return st.one_of(
        st.just({}),
        st.dictionaries(
            st.just(DEFAULT_DB_ALIAS),
            db_config,
            min_size=1,
            max_size=1
        ),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            db_config,
            max_size=5
        ).map(lambda d: {**d, DEFAULT_DB_ALIAS: d.get(DEFAULT_DB_ALIAS, {})})
    )


@given(make_database_config_strategy())
@settings(max_examples=500)
def test_configure_settings_sets_all_required_defaults(databases):
    handler = ConnectionHandler()
    result = handler.configure_settings(databases)

    required_keys = ['ENGINE', 'ATOMIC_REQUESTS', 'AUTOCOMMIT', 'CONN_MAX_AGE',
                     'CONN_HEALTH_CHECKS', 'OPTIONS', 'TIME_ZONE',
                     'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT', 'TEST']

    for db_config in result.values():
        for key in required_keys:
            assert key in db_config
```

**Failing input**: `databases={'default': {'TEST': ''}}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
result = handler.configure_settings(databases)
```

**Output**:
```
AttributeError: 'str' object has no attribute 'setdefault'
```

The error occurs at line 181 in `/django/db/utils.py`:
```python
test_settings = conn.setdefault("TEST", {})  # Line 172
# ...
for key, value in default_test_settings:
    test_settings.setdefault(key, value)  # Line 181 - crashes here
```

## Why This Is A Bug

The `configure_settings` method assumes that the `TEST` configuration value, if provided, will always be a dictionary. However, there is no validation to ensure this. When a user provides an invalid value like an empty string, integer, or boolean for `TEST`, the code crashes with a confusing `AttributeError` instead of providing a clear validation error.

While `TEST` being a non-dict is technically invalid configuration, the function should either:
1. Validate that `TEST` is a dict and raise a clear error message if it's not
2. Replace non-dict `TEST` values with an empty dict to handle gracefully

The current behavior provides a poor user experience with an unclear error message.

## Fix

Replace the non-dict TEST value with an empty dict before trying to configure test settings:

```diff
--- a/django/db/utils.py
+++ b/django/db/utils.py
@@ -169,7 +169,10 @@ class ConnectionHandler(BaseConnectionHandler):
             for setting in ["NAME", "USER", "PASSWORD", "HOST", "PORT"]:
                 conn.setdefault(setting, "")

-            test_settings = conn.setdefault("TEST", {})
+            # Ensure TEST is always a dict, replacing non-dict values
+            if not isinstance(conn.get("TEST"), dict):
+                conn["TEST"] = {}
+            test_settings = conn["TEST"]
             default_test_settings = [
                 ("CHARSET", None),
                 ("COLLATION", None),