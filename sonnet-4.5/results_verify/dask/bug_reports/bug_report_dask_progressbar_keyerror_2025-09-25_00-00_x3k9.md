# Bug Report: dask.diagnostics.progress.ProgressBar Missing State Keys

**Target**: `dask.diagnostics.progress.ProgressBar._update_bar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ProgressBar._update_bar` method crashes with a `KeyError` when the state dictionary is missing required keys (`"finished"`, `"ready"`, `"waiting"`, or `"running"`), even though it performs defensive checks for empty state dictionaries.

## Property-Based Test

```python
import sys
from io import StringIO

from hypothesis import given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.progress import ProgressBar


@given(st.sets(
    st.sampled_from(["finished", "ready", "waiting", "running"]),
    min_size=1,
    max_size=3
))
def test_progressbar_update_crashes_with_incomplete_state(present_keys):
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = {key: [] for key in present_keys}

    required_keys = {"finished", "ready", "waiting", "running"}
    missing_keys = required_keys - present_keys

    if missing_keys:
        try:
            pbar._update_bar(elapsed=1.0)
            assert False, f"Expected KeyError for missing keys: {missing_keys}"
        except KeyError as e:
            assert str(e).strip("'") in missing_keys
    else:
        pbar._update_bar(elapsed=1.0)
```

**Failing input**: State dict missing any of the required keys, e.g., `{"finished": [], "waiting": [], "running": []}` (missing "ready")

## Reproducing the Bug

```python
import sys
from io import StringIO

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.progress import ProgressBar

pbar = ProgressBar(out=StringIO())
pbar._start_time = 0
pbar._state = {
    "finished": ["task1", "task2"],
    "waiting": [],
    "running": []
}

pbar._update_bar(elapsed=1.0)
```

Output:
```
KeyError: 'ready'
```

## Why This Is A Bug

The code already performs defensive programming by checking `if not s:` at line 133, which handles completely empty state dictionaries. However, it fails to validate that all required keys are present when the state is non-empty. This creates an inconsistency where the function is defensive against some edge cases but not others.

While normal dask schedulers (see `dask/local.py:222-232`) create state dicts with all required keys, the code doesn't validate this assumption, making it fragile to:
- Custom schedulers with different state structures
- Callbacks that modify or delete state keys
- Direct manipulation of `ProgressBar._state`

The defensive check `if not s:` suggests the developers were aware of edge cases, but the implementation is incomplete.

## Fix

```diff
--- a/dask/diagnostics/progress.py
+++ b/dask/diagnostics/progress.py
@@ -131,10 +131,14 @@ class ProgressBar(Callback):
     def _update_bar(self, elapsed):
         s = self._state
         if not s:
             self._draw_bar(0, elapsed)
             return
-        ndone = len(s["finished"])
-        ntasks = sum(len(s[k]) for k in ["ready", "waiting", "running"]) + ndone
+        ndone = len(s.get("finished", []))
+        ntasks = sum(
+            len(s.get(k, []))
+            for k in ["ready", "waiting", "running"]
+        ) + ndone
         if ndone < ntasks:
             self._draw_bar(ndone / ntasks if ntasks else 0, elapsed)
```