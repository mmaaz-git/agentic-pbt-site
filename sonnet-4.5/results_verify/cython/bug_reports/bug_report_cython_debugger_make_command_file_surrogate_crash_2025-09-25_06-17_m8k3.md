# Bug Report: Cython.Debugger.Cygdb make_command_file Surrogate Character Crash

**Target**: `Cython.Debugger.Cygdb.make_command_file`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `make_command_file` function crashes with `UnicodeEncodeError` when the `prefix_code` parameter contains Unicode surrogate characters (U+D800 to U+DFFF), which are invalid in UTF-8.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import os
from hypothesis import given, strategies as st, settings
from Cython.Debugger.Cygdb import make_command_file


@settings(max_examples=100)
@given(st.text(alphabet=st.characters(whitelist_categories=('Cs',)), min_size=1, max_size=10))
def test_surrogate_characters_cause_crash(prefix_code):
    result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)
    try:
        with open(result, 'r') as f:
            f.read()
    finally:
        if os.path.exists(result):
            os.remove(result)
```

**Failing input**: `'\ud800'` (any surrogate character in the range U+D800 to U+DFFF)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.Cygdb import make_command_file

prefix_code = '\ud800'

try:
    result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)
    print("No crash - unexpected")
except UnicodeEncodeError as e:
    print(f"Bug confirmed: {e}")
```

## Why This Is A Bug

The function opens a temporary file in text mode (`os.fdopen(fd, 'w')`) and attempts to write `prefix_code` to it. When `prefix_code` contains Unicode surrogate characters, Python's UTF-8 encoder raises `UnicodeEncodeError` because surrogates are not valid in UTF-8.

While surrogate characters are technically invalid and should not appear in normal Python strings, the function should either:
1. Handle this case gracefully with a clear error message
2. Validate the input and reject surrogates explicitly
3. Document that `prefix_code` must be valid UTF-8

Currently, it crashes with an uninformative error that doesn't explain the problem to users.

## Fix

Add input validation to provide a clearer error message:

```diff
--- a/Cython/Debugger/Cygdb.py
+++ b/Cython/Debugger/Cygdb.py
@@ -26,6 +26,11 @@ def make_command_file(path_to_debug_info, prefix_code='',
                       no_import=False, skip_interpreter=False):
+    # Validate prefix_code doesn't contain surrogate characters
+    try:
+        prefix_code.encode('utf-8')
+    except UnicodeEncodeError:
+        raise ValueError("prefix_code contains invalid Unicode (surrogate characters)")
+
     if not no_import:
         pattern = os.path.join(path_to_debug_info,
                                'cython_debug',
```

Alternatively, use error handling with `errors='surrogatepass'` or `errors='replace'` when writing, though validation with a clear error message is preferable for API clarity.