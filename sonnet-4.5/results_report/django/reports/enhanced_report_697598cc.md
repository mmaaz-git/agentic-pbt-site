# Bug Report: django.apps.config.AppConfig.create IndexError on Trailing Dot

**Target**: `django.apps.config.AppConfig.create`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`AppConfig.create()` crashes with an unhelpful `IndexError: string index out of range` when called with an app configuration entry that ends with a dot (e.g., `"django.contrib.auth."`), instead of raising a clear error message explaining the invalid configuration.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Django AppConfig.create() trailing dot bug."""

import os
import sys
import django
from django.conf import settings
from hypothesis import given, strategies as st, settings as hypo_settings, HealthCheck, seed
import traceback

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        }
    )

# Setup Django
django.setup()

from django.apps.config import AppConfig

@given(st.text(min_size=1).filter(lambda s: '.' in s))
@hypo_settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
@seed(54184995177019238848460567174500504572)
def test_create_with_module_paths(entry):
    """Test that AppConfig.create() handles various module path formats gracefully."""
    if entry.endswith('.') and not entry.rpartition('.')[2]:
        # This should trigger an IndexError due to the bug
        try:
            AppConfig.create(entry)
            # If we get here without exception, that's unexpected but not a failure for this specific bug
            pass
        except IndexError as e:
            # This is the bug - we get IndexError instead of a proper error
            print(f"Found IndexError for input: {repr(entry)}")
            print(f"  Error message: {e}")
            # Re-raise to let Hypothesis know this is a failure
            raise
        except (ImportError, Exception):
            # Other exceptions are expected for invalid module paths
            pass
    else:
        # For non-trailing-dot entries, we expect either success or a proper exception
        try:
            AppConfig.create(entry)
        except IndexError:
            # IndexError should not happen for non-trailing-dot entries
            print(f"Unexpected IndexError for input: {repr(entry)}")
            raise
        except (ImportError, Exception):
            # Other exceptions are fine
            pass

if __name__ == "__main__":
    print("Running property-based test for AppConfig.create() trailing dot bug...")
    print("=" * 60)

    try:
        test_create_with_module_paths()
        print("Test completed without finding the bug (unexpected)")
    except Exception as e:
        print("\nTest failed (bug found)!")
        print("-" * 60)
        print("Full traceback:")
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `'0.'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 69, in <module>
    test_create_with_module_paths()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 33, in test_create_with_module_paths
    @hypo_settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 40, in test_create_with_module_paths
    AppConfig.create(entry)
    ~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/apps/config.py", line 172, in create
    if mod_path and cls_name[0].isupper():
                    ~~~~~~~~^^^
IndexError: string index out of range
Falsifying example: test_create_with_module_paths(
    entry='0.',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/10/hypo.py:39
Running property-based test for AppConfig.create() trailing dot bug...
============================================================
Found IndexError for input: 'T\x11Ëá.'
  Error message: string index out of range
Found IndexError for input: 'T\x11Ëá.'
  Error message: string index out of range
Found IndexError for input: 'T\x11Ëá.'
  Error message: string index out of range
Found IndexError for input: 'T\x11Ë.'
  Error message: string index out of range
Found IndexError for input: 'T\x11.'
  Error message: string index out of range
Found IndexError for input: 'T.'
  Error message: string index out of range
Found IndexError for input: '0.'
  Error message: string index out of range
Found IndexError for input: '0.'
  Error message: string index out of range

Test failed (bug found)!
------------------------------------------------------------
Full traceback:
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Django AppConfig.create() IndexError bug."""

import os
import sys
import django
from django.conf import settings

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        }
    )

# Setup Django
django.setup()

# Now reproduce the bug
from django.apps.config import AppConfig

try:
    # This should raise an IndexError instead of a proper error message
    result = AppConfig.create("django.contrib.auth.")
    print(f"Unexpectedly succeeded: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception ({type(e).__name__}): {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
IndexError: string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/repo.py", line 33, in <module>
    result = AppConfig.create("django.contrib.auth.")
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/apps/config.py", line 172, in create
    if mod_path and cls_name[0].isupper():
                    ~~~~~~~~^^^
IndexError: string index out of range
IndexError: string index out of range
```
</details>

## Why This Is A Bug

The bug occurs in `/django/apps/config.py` at lines 171-172 when processing an entry string that ends with a dot:

```python
mod_path, _, cls_name = entry.rpartition(".")
if mod_path and cls_name[0].isupper():
```

When `entry = "django.contrib.auth."` (or any string ending with a dot):
1. `entry.rpartition(".")` returns `('django.contrib.auth', '.', '')`
2. This sets `mod_path = 'django.contrib.auth'` (which is truthy) and `cls_name = ''` (an empty string)
3. The code then attempts to access `cls_name[0]` without checking if `cls_name` is empty
4. Accessing index 0 on an empty string raises `IndexError: string index out of range`

This violates Django's documented intent to provide helpful error messages. The code comments at line 170 explicitly state: "Provide a nice error message in both cases" - but an IndexError is not a nice error message. The method should either handle the input gracefully or raise a meaningful exception like `ImproperlyConfigured` with a clear message about what's wrong.

## Relevant Context

The `AppConfig.create()` method is a factory that creates an app config from an entry in Django's `INSTALLED_APPS` setting. While this method appears to be an internal API (it's not documented in Django's public documentation), it's still used by Django's app loading mechanism.

The bug manifests when:
- A user accidentally adds a trailing dot to an INSTALLED_APPS entry
- Programmatic string manipulation results in a trailing dot
- Copy-paste errors introduce trailing dots

Django's code already handles other invalid inputs gracefully:
- Invalid module paths trigger `ImportError` with helpful messages
- Invalid class names trigger `ImproperlyConfigured` with suggestions
- The code specifically aims to provide "nice error messages" for configuration mistakes

Source code location: [django/apps/config.py line 172](https://github.com/django/django/blob/main/django/apps/config.py#L172)

## Proposed Fix

```diff
--- a/django/apps/config.py
+++ b/django/apps/config.py
@@ -169,7 +169,7 @@ class AppConfig:
             # then it was likely intended to be an app config class; if not,
             # an app module. Provide a nice error message in both cases.
             mod_path, _, cls_name = entry.rpartition(".")
-            if mod_path and cls_name[0].isupper():
+            if mod_path and cls_name and cls_name[0].isupper():
                 # We could simply re-trigger the string import exception, but
                 # we're going the extra mile and providing a better error
                 # message for typos in INSTALLED_APPS.
```