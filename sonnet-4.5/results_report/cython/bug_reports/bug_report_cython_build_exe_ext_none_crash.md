# Bug Report: Cython.Build.BuildExecutable EXE_EXT Can Be None Causing TypeError

**Target**: `Cython.Build.BuildExecutable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The module-level variable `EXE_EXT` is assigned from `sysconfig.get_config_var('EXE')`, which can return `None` on some platforms. When `EXE_EXT` is `None`, string concatenation operations like `basename + EXE_EXT` raise `TypeError`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import sysconfig


@given(st.text(min_size=1, max_size=20))
def test_exe_ext_string_concatenation(basename):
    import importlib
    original_get_config_var = sysconfig.get_config_var

    def mock_get_config_var(name):
        if name == 'EXE':
            return None
        return original_get_config_var(name)

    sysconfig.get_config_var = mock_get_config_var

    try:
        if 'Cython.Build.BuildExecutable' in sys.modules:
            importlib.reload(sys.modules['Cython.Build.BuildExecutable'])
        else:
            import Cython.Build.BuildExecutable

        from Cython.Build.BuildExecutable import EXE_EXT
        result = basename + EXE_EXT

    finally:
        sysconfig.get_config_var = original_get_config_var
```

**Failing input**: Any basename string when `sysconfig.get_config_var('EXE')` returns `None`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import sysconfig

original_get_config_var = sysconfig.get_config_var

def mock_get_config_var(name):
    if name == 'EXE':
        return None
    return original_get_config_var(name)

sysconfig.get_config_var = mock_get_config_var

try:
    import importlib
    import Cython.Build.BuildExecutable
    importlib.reload(Cython.Build.BuildExecutable)

    from Cython.Build.BuildExecutable import EXE_EXT, build

    print(f"EXE_EXT value: {EXE_EXT!r}")

    basename = "test_program"
    exe_name = basename + EXE_EXT

except TypeError as e:
    print(f"BUG: {e}")
    print("Cannot concatenate string with None")

finally:
    sysconfig.get_config_var = original_get_config_var
```

## Why This Is A Bug

When `sysconfig.get_config_var('EXE')` returns `None` (which can happen on platforms where the EXE extension variable is not set), the module initialization sets `EXE_EXT = None`. Subsequent string operations throughout the module (lines 110, 139) attempt to concatenate strings with `EXE_EXT`, causing `TypeError: can only concatenate str (not "NoneType") to str`.

## Fix

```diff
--- a/Cython/Build/BuildExecutable.py
+++ b/Cython/Build/BuildExecutable.py
@@ -48,7 +48,7 @@ LINKCC = get_config_var('LINKCC', os.environ.get('LINKCC', CC))
 LINKFORSHARED = get_config_var('LINKFORSHARED')
 LIBS = get_config_var('LIBS')
 SYSLIBS = get_config_var('SYSLIBS')
-EXE_EXT = sysconfig.get_config_var('EXE')
+EXE_EXT = sysconfig.get_config_var('EXE') or ''


 def _debug(msg, *args):
```