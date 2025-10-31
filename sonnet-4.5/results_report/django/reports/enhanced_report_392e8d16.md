# Bug Report: django.db.utils.ConnectionHandler.configure_settings TEST Setting Type Validation Failure

**Target**: `django.db.utils.ConnectionHandler.configure_settings`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `configure_settings` method in Django's `ConnectionHandler` crashes with an `AttributeError` when the `TEST` database configuration setting is provided as a non-dict value (e.g., string, integer, boolean), as it assumes `TEST` is always a dictionary without performing type validation.

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

if __name__ == "__main__":
    test_configure_settings_sets_all_required_defaults()
```

<details>

<summary>
**Failing input**: `databases={'default': {'TEST': ''}}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 54, in <module>
    test_configure_settings_sets_all_required_defaults()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 40, in test_configure_settings_sets_all_required_defaults
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 43, in test_configure_settings_sets_all_required_defaults
    result = handler.configure_settings(databases)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/utils.py", line 181, in configure_settings
    test_settings.setdefault(key, value)
    ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'setdefault'
Falsifying example: test_configure_settings_sets_all_required_defaults(
    databases={'default': {'TEST': ''}},
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.utils import ConnectionHandler

handler = ConnectionHandler()
databases = {'default': {'TEST': ''}}
result = handler.configure_settings(databases)
```

<details>

<summary>
AttributeError: 'str' object has no attribute 'setdefault'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/repo.py", line 8, in <module>
    result = handler.configure_settings(databases)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/utils.py", line 181, in configure_settings
    test_settings.setdefault(key, value)
    ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'setdefault'
```
</details>

## Why This Is A Bug

This violates expected behavior because the `configure_settings` method is responsible for validating and normalizing database configurations, yet it fails to handle invalid input gracefully. The method uses `conn.setdefault("TEST", {})` on line 172, which returns the existing value if the "TEST" key already exists, regardless of its type. When TEST contains a non-dict value (like an empty string), the subsequent call to `test_settings.setdefault(key, value)` on line 181 fails because non-dict objects don't have a `setdefault` method.

The error message `AttributeError: 'str' object has no attribute 'setdefault'` provides no guidance about the actual problem - that TEST must be a dictionary. This creates a poor developer experience, especially for users new to Django who might misconfigure their database settings.

## Relevant Context

The `configure_settings` method in `/django/db/utils.py` (lines 147-182) is responsible for setting default values for database configuration. It correctly handles other configuration values by using `setdefault()` to provide defaults, but fails to validate that TEST is a dictionary before attempting to set default values within it.

Throughout the Django codebase, TEST is consistently accessed as a dictionary:
- `self.connection.settings_dict["TEST"]["NAME"]` in base/creation.py
- `self.connection.settings_dict["TEST"]["MIGRATE"]` in multiple locations
- Backend-specific code in mysql/creation.py and postgresql/creation.py expects TEST to be a dict

The expected TEST dictionary structure should contain these optional keys with defaults:
- CHARSET (default: None)
- COLLATION (default: None)
- MIGRATE (default: True)
- MIRROR (default: None)
- NAME (default: None)

## Proposed Fix

Replace non-dict TEST values with an empty dict before attempting to set defaults:

```diff
--- a/django/db/utils.py
+++ b/django/db/utils.py
@@ -169,7 +169,11 @@ class ConnectionHandler(BaseConnectionHandler):
             for setting in ["NAME", "USER", "PASSWORD", "HOST", "PORT"]:
                 conn.setdefault(setting, "")

-            test_settings = conn.setdefault("TEST", {})
+            # Ensure TEST is always a dict, replacing invalid values
+            if "TEST" in conn and not isinstance(conn["TEST"], dict):
+                conn["TEST"] = {}
+            test_settings = conn.setdefault("TEST", {})
+
             default_test_settings = [
                 ("CHARSET", None),
                 ("COLLATION", None),
```