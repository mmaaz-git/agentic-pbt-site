# Bug Report: django.conf Settings TIME_ZONE Path Traversal

**Target**: `django.conf.Settings.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's Settings class accepts TIME_ZONE values with path traversal sequences (e.g., `../../../tmp/file`) that escape the intended `/usr/share/zoneinfo` directory. The timezone validation logic only checks if the constructed path exists, not whether it's actually within the zoneinfo directory.

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
```

**Failing input**: A TIME_ZONE value like `../../../tmp/tmp7mmc6kfh/fake_timezone` that points to an existing file outside `/usr/share/zoneinfo`

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

## Why This Is A Bug

The TIME_ZONE validation logic at lines 195-205 in `django/conf/__init__.py` constructs a path and only checks if it exists:

```python
if hasattr(time, "tzset") and self.TIME_ZONE:
    zoneinfo_root = Path("/usr/share/zoneinfo")
    zone_info_file = zoneinfo_root.joinpath(*self.TIME_ZONE.split("/"))
    if zoneinfo_root.exists() and not zone_info_file.exists():
        raise ValueError("Incorrect timezone setting: %s" % self.TIME_ZONE)
```

The check `if ... not zone_info_file.exists()` only raises an error if the file doesn't exist. If a TIME_ZONE with path traversal sequences (like `../../../tmp/file`) points to an existing file, Django accepts it even though it's outside the intended directory.

This violates the implicit contract that timezone files should come from the system's zoneinfo directory. While not a critical security vulnerability (since the attacker would need to control the settings file), it represents unexpected behavior that could lead to:
1. Confusion about which timezone data is being used
2. Potential security issues in specific deployment scenarios
3. Violation of principle of least surprise

## Fix

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -197,8 +197,15 @@ class Settings:
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
```