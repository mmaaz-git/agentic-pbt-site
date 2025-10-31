# Bug Report: django.conf.Settings TIME_ZONE Path Traversal Vulnerability

**Target**: `django.conf.Settings.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's Settings class accepts TIME_ZONE values containing path traversal sequences (e.g., `../../../tmp/file`) that allow timezone files to be loaded from arbitrary locations outside the intended `/usr/share/zoneinfo` directory, violating security best practices and IANA timezone naming conventions.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st
from django.conf import Settings


def test_timezone_path_should_stay_within_zoneinfo_root():
    zoneinfo_root = Path("/usr/share/zoneinfo")

    if not zoneinfo_root.exists():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        external_file = Path(tmpdir) / "fake_timezone"
        external_file.touch()

        relative_path = os.path.relpath(external_file, zoneinfo_root)

        parts = relative_path.split("/")
        zone_info_file = zoneinfo_root.joinpath(*parts)

        is_inside = (
            zoneinfo_root.resolve() == zone_info_file.resolve() or
            zoneinfo_root.resolve() in zone_info_file.resolve().parents
        )

        if zone_info_file.exists() and not is_inside:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"TIME_ZONE = {repr(relative_path)}\n")
                f.write("SECRET_KEY = 'test'\n")
                settings_file = f.name

            try:
                module_name = Path(settings_file).stem
                sys.path.insert(0, str(Path(settings_file).parent))

                try:
                    settings_obj = Settings(module_name)
                    assert False, "Django accepted timezone path outside zoneinfo_root"
                except ValueError:
                    pass
                finally:
                    sys.path.remove(str(Path(settings_file).parent))
                    if module_name in sys.modules:
                        del sys.modules[module_name]
            finally:
                os.unlink(settings_file)

if __name__ == "__main__":
    test_timezone_path_should_stay_within_zoneinfo_root()
```

<details>

<summary>
**Failing input**: `../../../tmp/tmptrsppzyh/fake_timezone`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 54, in <module>
    test_timezone_path_should_stay_within_zoneinfo_root()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 43, in test_timezone_path_should_stay_within_zoneinfo_root
    assert False, "Django accepted timezone path outside zoneinfo_root"
           ^^^^^
AssertionError: Django accepted timezone path outside zoneinfo_root
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
import tempfile
from pathlib import Path
from django.conf import Settings

zoneinfo_root = Path("/usr/share/zoneinfo")

with tempfile.TemporaryDirectory() as tmpdir:
    external_file = Path(tmpdir) / "fake_timezone"
    external_file.touch()

    relative_path = os.path.relpath(external_file, zoneinfo_root)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f"TIME_ZONE = {repr(relative_path)}\n")
        f.write("SECRET_KEY = 'test'\n")
        settings_file = f.name

    try:
        module_name = Path(settings_file).stem
        sys.path.insert(0, str(Path(settings_file).parent))

        try:
            settings_obj = Settings(module_name)
            print(f"BUG: Django accepted TIME_ZONE = {repr(relative_path)}")
            print(f"This path points to: {zoneinfo_root.joinpath(*relative_path.split('/')).resolve()}")
            print(f"Which is outside: {zoneinfo_root.resolve()}")
        finally:
            sys.path.remove(str(Path(settings_file).parent))
            if module_name in sys.modules:
                del sys.modules[module_name]
    finally:
        os.unlink(settings_file)
```

<details>

<summary>
Django accepts path traversal in TIME_ZONE setting
</summary>
```
BUG: Django accepted TIME_ZONE = '../../../tmp/tmptrsppzyh/fake_timezone'
This path points to: /tmp/tmptrsppzyh/fake_timezone
Which is outside: /usr/share/zoneinfo
```
</details>

## Why This Is A Bug

This bug violates multiple expected behaviors and security principles:

1. **Path Traversal Security Vulnerability (CWE-22)**: The Settings class fails to prevent path traversal sequences in the TIME_ZONE configuration, allowing timezone data to be loaded from arbitrary files on the filesystem. While exploitation requires control of the Django settings file (which already provides significant access), this still violates the defense-in-depth security principle.

2. **Broken Validation Logic**: The timezone validation code at lines 195-205 in `django/conf/__init__.py` explicitly attempts to validate timezone files within `/usr/share/zoneinfo`, but the validation is incomplete. The code constructs a path using `zoneinfo_root.joinpath(*self.TIME_ZONE.split("/"))` and only checks if the file exists (`if zoneinfo_root.exists() and not zone_info_file.exists()`), not whether it's actually within the zoneinfo directory.

3. **IANA Standard Violation**: The IANA Time Zone Database defines standard timezone identifiers like "America/New_York" and "Europe/Paris" that correspond to files within the system's zoneinfo directory. Path traversal sequences like "../../../" are not valid IANA timezone identifiers and should never be accepted.

4. **Documentation Mismatch**: Django's documentation recommends using standard IANA timezone names and suggests using `zoneinfo.available_timezones()` to get valid keys. Accepting path traversal sequences contradicts this documented behavior.

5. **Principle of Least Surprise**: No reasonable developer would expect `TIME_ZONE = "../../../etc/passwd"` to be accepted as a valid timezone setting. This unexpected behavior could lead to confusion about which timezone data is being used and potential security issues in multi-tenant or containerized environments.

## Relevant Context

The vulnerable code is located at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/conf/__init__.py` lines 195-205:

```python
if hasattr(time, "tzset") and self.TIME_ZONE:
    # When we can, attempt to validate the timezone. If we can't find
    # this file, no check happens and it's harmless.
    zoneinfo_root = Path("/usr/share/zoneinfo")
    zone_info_file = zoneinfo_root.joinpath(*self.TIME_ZONE.split("/"))
    if zoneinfo_root.exists() and not zone_info_file.exists():
        raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
    # Move the time zone info into os.environ. See ticket #2315 for why
    # we don't do this unconditionally (breaks Windows).
    os.environ["TZ"] = self.TIME_ZONE
    time.tzset()
```

The code comment states "it's harmless" when validation doesn't happen, but this refers to cases where `/usr/share/zoneinfo` doesn't exist (e.g., on Windows), not to accepting arbitrary file paths through path traversal.

Django documentation on TIME_ZONE setting: https://docs.djangoproject.com/en/stable/ref/settings/#std-setting-TIME_ZONE

IANA Time Zone Database: https://www.iana.org/time-zones

CWE-22 (Path Traversal): https://cwe.mitre.org/data/definitions/22.html

## Proposed Fix

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -195,11 +195,18 @@ class Settings:
         if hasattr(time, "tzset") and self.TIME_ZONE:
             # When we can, attempt to validate the timezone. If we can't find
             # this file, no check happens and it's harmless.
             zoneinfo_root = Path("/usr/share/zoneinfo")
             zone_info_file = zoneinfo_root.joinpath(*self.TIME_ZONE.split("/"))
-            if zoneinfo_root.exists() and not zone_info_file.exists():
-                raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
+            if zoneinfo_root.exists():
+                # Resolve both paths to check if zone_info_file is actually inside zoneinfo_root
+                resolved_root = zoneinfo_root.resolve()
+                resolved_zone = zone_info_file.resolve()
+                is_inside = (resolved_root == resolved_zone or
+                           resolved_root in resolved_zone.parents)
+
+                if not zone_info_file.exists() or not is_inside:
+                    raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
             # Move the time zone info into os.environ. See ticket #2315 for why
             # we don't do this unconditionally (breaks Windows).
             os.environ["TZ"] = self.TIME_ZONE
             time.tzset()
```