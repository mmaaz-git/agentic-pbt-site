# Bug Report: pydantic.plugin._loader.get_plugins Case-Sensitive Environment Variable Check

**Target**: `pydantic.plugin._loader.get_plugins`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `PYDANTIC_DISABLE_PLUGINS` environment variable check is case-sensitive, only recognizing lowercase 'true' but not common variations like 'True' or 'TRUE', violating standard environment variable conventions.

## Property-Based Test

```python
from hypothesis import given, strategies as st


@given(st.sampled_from(['true', 'True', 'TRUE', '1', '__all__']))
def test_disable_all_plugins_case_insensitive(value):
    disabled_plugins = value

    is_truthy = value.lower() in ('true', '1', '__all__')

    actual_disabled = disabled_plugins in ('__all__', '1', 'true')

    assert actual_disabled == is_truthy, (
        f"PYDANTIC_DISABLE_PLUGINS='{value}' should disable all plugins "
        f"regardless of case, but case-sensitive check fails"
    )


if __name__ == '__main__':
    # Run the test
    test_disable_all_plugins_case_insensitive()
```

<details>

<summary>
**Failing input**: `'True'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 20, in <module>
    test_disable_all_plugins_case_insensitive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 5, in test_disable_all_plugins_case_insensitive
    def test_disable_all_plugins_case_insensitive(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 12, in test_disable_all_plugins_case_insensitive
    assert actual_disabled == is_truthy, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: PYDANTIC_DISABLE_PLUGINS='True' should disable all plugins regardless of case, but case-sensitive check fails
Falsifying example: test_disable_all_plugins_case_insensitive(
    value='True',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/7/hypo.py:13
```
</details>

## Reproducing the Bug

```python
import os

# Test the case-sensitivity issue with PYDANTIC_DISABLE_PLUGINS
test_values = ['true', 'True', 'TRUE', '1', '__all__']

print("Testing PYDANTIC_DISABLE_PLUGINS case sensitivity:\n")

for value in test_values:
    # Simulate the exact check from pydantic/plugin/_loader.py line 32
    disabled_plugins = value

    if disabled_plugins in ('__all__', '1', 'true'):
        result = "DISABLES plugins"
    else:
        result = "DOES NOT disable plugins"

    # What users would reasonably expect
    expected = "DISABLES" if value.lower() in ('true', '1', '__all__') else "DOES NOT disable"

    match = "✓" if result.startswith(expected) else "✗"
    print(f"  {match} PYDANTIC_DISABLE_PLUGINS='{value}': {result}")

    if not result.startswith(expected):
        print(f"     Expected: {expected} plugins (case-insensitive check)")

print("\nBug: 'True' and 'TRUE' do not disable plugins despite being reasonable values.")
```

<details>

<summary>
Case-sensitive check fails for capitalized boolean values
</summary>
```
Testing PYDANTIC_DISABLE_PLUGINS case sensitivity:

  ✓ PYDANTIC_DISABLE_PLUGINS='true': DISABLES plugins
  ✗ PYDANTIC_DISABLE_PLUGINS='True': DOES NOT disable plugins
     Expected: DISABLES plugins (case-insensitive check)
  ✗ PYDANTIC_DISABLE_PLUGINS='TRUE': DOES NOT disable plugins
     Expected: DISABLES plugins (case-insensitive check)
  ✓ PYDANTIC_DISABLE_PLUGINS='1': DISABLES plugins
  ✓ PYDANTIC_DISABLE_PLUGINS='__all__': DISABLES plugins

Bug: 'True' and 'TRUE' do not disable plugins despite being reasonable values.
```
</details>

## Why This Is A Bug

The case-sensitive check on line 32 of `/pydantic/plugin/_loader.py` violates standard environment variable conventions where boolean values are typically case-insensitive. The code performs an exact string match `disabled_plugins in ('__all__', '1', 'true')` which fails for 'True' and 'TRUE'.

This contradicts:
1. **Common practice**: Most environment variable parsers accept 'true', 'True', 'TRUE' interchangeably
2. **User expectations**: Configuration tools (Docker Compose, Ansible, CI/CD systems) often use 'True' with capital T
3. **Python conventions**: `str(True)` returns 'True' with capital T, making it a natural choice
4. **Documentation ambiguity**: The Pydantic documentation shows 'true' in the allowed values table but never explicitly states that the check is case-sensitive

The result is silent failure - users who set `PYDANTIC_DISABLE_PLUGINS=True` believe they've disabled plugins, but plugins remain active, causing confusion and debugging time.

## Relevant Context

- **Source location**: `/pydantic/plugin/_loader.py`, line 32
- **Documentation**: https://docs.pydantic.dev/2.8/concepts/plugins/ - shows allowed values but doesn't specify case sensitivity
- **Environment variable conventions**: Most tools (Docker, Ansible, GitHub Actions) treat boolean env vars case-insensitively
- **Python behavior**: `bool('True')` and `bool('true')` both return `True`, setting expectation for case-insensitive handling
- **Shell scripting**: Common to use `export PYDANTIC_DISABLE_PLUGINS=True` from Python scripts or YAML configs

## Proposed Fix

```diff
--- a/pydantic/plugin/_loader.py
+++ b/pydantic/plugin/_loader.py
@@ -29,7 +29,7 @@ def get_plugins() -> Iterable[PydanticPluginProtocol]:
     if _loading_plugins:
         # this happens when plugins themselves use pydantic, we return no plugins
         return ()
-    elif disabled_plugins in ('__all__', '1', 'true'):
+    elif disabled_plugins and disabled_plugins.lower() in ('__all__', '1', 'true'):
         return ()
     elif _plugins is None:
         _plugins = {}
```