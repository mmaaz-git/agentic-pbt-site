# Bug Report: Jinja2Templates.__init__ Empty Sequence Handling

**Target**: `starlette.templating.Jinja2Templates.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Jinja2Templates.__init__` method has inconsistent validation logic between its assertion and conditional checks. When an empty sequence (e.g., `[]` or `()`) is passed for the `directory` parameter alongside a valid `env` parameter, the assertion passes but the code incorrectly attempts to create an environment from the empty directory instead of using the provided `env`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import jinja2
from starlette.templating import Jinja2Templates

@given(st.sampled_from([[], (), ""]))
def test_jinja2templates_empty_directory_with_env(empty_directory):
    custom_env = jinja2.Environment()

    templates = Jinja2Templates(directory=empty_directory, env=custom_env)

    assert templates.env is custom_env, \
        f"Expected templates.env to be custom_env when directory={empty_directory!r}, but got a different env"
```

**Failing input**: `directory=[]` (or `()` or `""`) with a valid `env` object

## Reproducing the Bug

```python
import jinja2
from starlette.templating import Jinja2Templates

custom_env = jinja2.Environment()

templates = Jinja2Templates(directory=[], env=custom_env)

print(f"templates.env is custom_env: {templates.env is custom_env}")

assert templates.env is custom_env, "Expected custom_env to be used, but a new env was created instead"
```

## Why This Is A Bug

The bug exists in `/starlette/templating.py` at lines 98-103:

```python
assert bool(directory) ^ bool(env), "either 'directory' or 'env' arguments must be passed"
self.context_processors = context_processors or []
if directory is not None:
    self.env = self._create_env(directory, **env_options)
elif env is not None:
    self.env = env
```

The assertion checks `bool(directory) ^ bool(env)` (XOR on truthiness), while the conditional check uses `directory is not None`. This inconsistency creates the following scenario:

- `directory = []`, `env = <valid Environment>`
- `bool([])` = `False`, `bool(env)` = `True`
- `False ^ True` = `True` → assertion passes ✓
- `[] is not None` = `True` → creates env from empty directory ✗

The code should use the provided `env` but instead attempts to create a new environment from the empty directory list, which will likely fail or produce unexpected behavior.

## Fix

Change line 100 to use truthiness check consistent with the assertion:

```diff
--- a/starlette/templating.py
+++ b/starlette/templating.py
@@ -97,7 +97,7 @@ class Jinja2Templates:
         assert jinja2 is not None, "jinja2 must be installed to use Jinja2Templates"
         assert bool(directory) ^ bool(env), "either 'directory' or 'env' arguments must be passed"
         self.context_processors = context_processors or []
-        if directory is not None:
+        if directory:
             self.env = self._create_env(directory, **env_options)
         elif env is not None:  # pragma: no branch
             self.env = env
```

Alternatively, change the assertion to use `is not None` checks, but the current approach (truthiness) is more appropriate since empty sequences are not valid directory specifications.