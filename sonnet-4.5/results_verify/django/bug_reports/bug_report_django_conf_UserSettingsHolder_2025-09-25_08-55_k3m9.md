# Bug Report: django.conf.UserSettingsHolder Allows Lowercase Settings

**Target**: `django.conf.UserSettingsHolder`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`UserSettingsHolder.__setattr__` allows setting lowercase attribute names, violating the contract that Django settings must be uppercase. This bypasses the validation enforced by `LazySettings.configure()` and creates an API inconsistency.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.conf import UserSettingsHolder, global_settings


@given(st.text(min_size=1))
def test_usersettingsholder_uppercase_contract(setting_name):
    """
    UserSettingsHolder should enforce that settings are uppercase,
    consistent with LazySettings.configure()'s validation.
    """
    assume(setting_name.upper() != setting_name)
    assume(setting_name not in {'default_settings', 'SETTINGS_MODULE'})
    assume(not setting_name.startswith('_'))

    holder = UserSettingsHolder(global_settings)

    holder.__setattr__(setting_name, "test_value")

    try:
        holder.__getattribute__(setting_name)
        assert False, f"Should not allow setting/getting lowercase setting {setting_name!r}"
    except AttributeError:
        pass
```

**Failing input**: `setting_name='a'`

## Reproducing the Bug

```python
from django.conf import UserSettingsHolder, LazySettings, global_settings

holder = UserSettingsHolder(global_settings)
holder.my_setting = "test"
print(holder.my_setting)

settings = LazySettings()
try:
    settings.configure(my_setting="test")
except TypeError as e:
    print(f"configure() rejects lowercase: {e}")
```

Output:
```
test
configure() rejects lowercase: Setting 'my_setting' must be uppercase.
```

## Why This Is A Bug

1. **Contract violation**: Django's settings contract requires uppercase names. This is enforced in `LazySettings.configure()` (line 121-122):
   ```python
   if not name.isupper():
       raise TypeError("Setting %r must be uppercase." % name)
   ```

2. **API inconsistency**: `LazySettings.configure()` validates uppercase, but `UserSettingsHolder.__setattr__` doesn't, allowing users to bypass validation by using `UserSettingsHolder` directly.

3. **Partial enforcement**: `UserSettingsHolder.__getattr__` (line 232-235) only retrieves uppercase attributes from `default_settings`, but doesn't prevent lowercase attributes from being set and retrieved via `__dict__`.

4. **Confusing behavior**: The class documentation states it's a "Holder for user configured settings", implying it should follow the same rules as `configure()`.

## Fix

Add validation to `UserSettingsHolder.__setattr__` to enforce uppercase settings (excluding internal attributes):

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -236,6 +236,11 @@ class UserSettingsHolder:

     def __setattr__(self, name, value):
         self._deleted.discard(name)
+        # Enforce uppercase for settings (but allow internal attributes)
+        if not name.isupper() and name not in {'default_settings'} and not name.startswith('_'):
+            raise TypeError(
+                "Setting %r must be uppercase." % name
+            )
         if name == "FORMS_URLFIELD_ASSUME_HTTPS":
             warnings.warn(
                 FORMS_URLFIELD_ASSUME_HTTPS_DEPRECATED_MSG,
```