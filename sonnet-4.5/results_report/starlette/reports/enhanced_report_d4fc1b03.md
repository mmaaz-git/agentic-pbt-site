# Bug Report: starlette.templating.Jinja2Templates.__init__ Empty Sequence Handling Inconsistency

**Target**: `starlette.templating.Jinja2Templates.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Jinja2Templates.__init__` method contains inconsistent validation logic between its assertion check and conditional branches. When an empty sequence (e.g., `[]`, `()`, or `""`) is passed as the `directory` parameter alongside a valid `env` parameter, the assertion passes but the code incorrectly creates a new environment from the empty directory instead of using the provided `env`.

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

if __name__ == "__main__":
    test_jinja2templates_empty_directory_with_env()
```

<details>

<summary>
**Failing input**: `empty_directory=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 15, in <module>
    test_jinja2templates_empty_directory_with_env()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 6, in test_jinja2templates_empty_directory_with_env
    def test_jinja2templates_empty_directory_with_env(empty_directory):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 11, in test_jinja2templates_empty_directory_with_env
    assert templates.env is custom_env, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected templates.env to be custom_env when directory=[], but got a different env
Falsifying example: test_jinja2templates_empty_directory_with_env(
    empty_directory=[],
)
```
</details>

## Reproducing the Bug

```python
import jinja2
from starlette.templating import Jinja2Templates

custom_env = jinja2.Environment()

templates = Jinja2Templates(directory=[], env=custom_env)

print(f"templates.env is custom_env: {templates.env is custom_env}")

assert templates.env is custom_env, "Expected custom_env to be used, but a new env was created instead"
```

<details>

<summary>
AssertionError: Expected custom_env to be used, but a new env was created instead
</summary>
```
templates.env is custom_env: False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/repo.py", line 10, in <module>
    assert templates.env is custom_env, "Expected custom_env to be used, but a new env was created instead"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected custom_env to be used, but a new env was created instead
```
</details>

## Why This Is A Bug

This violates expected behavior due to an inconsistency between the assertion logic and the conditional branching logic in the `__init__` method. The bug manifests at lines 98-103 in `/starlette/templating.py`:

1. **Line 98** uses an XOR assertion: `assert bool(directory) ^ bool(env)` - This checks the truthiness of both parameters
2. **Line 100** uses a None check: `if directory is not None:` - This checks if directory is not None

For empty sequences like `[]`, `()`, or `""`:
- `bool([])` evaluates to `False` (empty sequences are falsy)
- `[] is not None` evaluates to `True` (empty list is not None)

This creates a logic error where:
- The assertion correctly interprets an empty sequence as "no directory provided" (falsy)
- The conditional incorrectly interprets an empty sequence as "directory provided" (not None)
- Result: When `directory=[]` and `env=custom_env`, the code creates a new environment instead of using the provided one

The assertion's intent is clear from its message: "either 'directory' or 'env' arguments must be passed" - emphasizing mutual exclusivity. The type hints via `@overload` decorators (lines 67-82) also support this: one overload accepts `directory` without `env`, the other accepts `env` without `directory`.

## Relevant Context

The Starlette documentation states that the directory parameter can be "a string, os.Pathlike or a list of strings or os.Pathlike denoting a directory path". An empty sequence doesn't represent a valid directory path, so the assertion's truthiness check is the correct interpretation.

When `_create_env` is called with an empty directory list (line 101), it passes this to `jinja2.FileSystemLoader(directory)` (line 112), which may create an environment with no template directories, losing any custom configuration from the user's provided environment.

Code location: `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/templating.py:98-103`

## Proposed Fix

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