# Bug Report: Cython.Plex Begin Action Crashes Due to Missing Scanner.begin() Method

**Target**: `Cython.Plex.Actions.Begin` and `Cython.Plex.Scanners.Scanner`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Begin` action, a core feature for state management in Plex lexical scanners, crashes with `AttributeError` because it internally calls `scanner.begin()` which doesn't exist on Scanner instances.

## Property-Based Test

```python
import io
from hypothesis import given, strategies as st, settings
from Cython.Plex import *

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=200)
def test_begin_action_changes_state(trigger_pattern):
    state1_pattern = 'x'
    state2_pattern = 'y'

    lexicon = Lexicon([
        (Str(trigger_pattern), Begin('state2')),
        State('state2', [
            (Str(state2_pattern), 'STATE2_TOKEN')
        ])
    ])

    scanner = Scanner(lexicon, io.StringIO(trigger_pattern + state2_pattern))

    token1, text1 = scanner.read()
    token2, text2 = scanner.read()
```

**Failing input**: `trigger_pattern='a'` (crashes on first `scanner.read()`)

## Reproducing the Bug

```python
import io
from Cython.Plex import *

lexicon = Lexicon([
    (Str('start'), Begin('state2')),
    State('state2', [(Str('x'), 'STATE2_X')])
])

scanner = Scanner(lexicon, io.StringIO('start'))

try:
    token, text = scanner.read()
    print(f'Success: {token!r}')
except AttributeError as e:
    print(f'AttributeError: {e}')
```

Output:
```
AttributeError: 'Cython.Plex.Scanners.Scanner' object has no attribute 'begin'
```

## Why This Is A Bug

This is a critical bug that makes multi-state lexical analysis impossible:

1. **The Begin action is documented as a core feature** - Both the Lexicon and Scanner docstrings describe state management as a primary feature of Plex.

2. **Begin.perform() calls scanner.begin()** - At `Actions.py:82`, the Begin action implementation calls:
   ```python
   token_stream.begin(self.state_name)
   ```

3. **Scanner doesn't implement begin()** - Inspecting Scanner reveals:
   - Public methods: `eof()`, `get_position()`, `position()`, `read()`
   - Missing: `begin()`, `produce()`

4. **State management is completely broken** - Any attempt to use the `Begin` action results in an immediate crash, making it impossible to implement lexers with multiple states (a fundamental feature of scanner design).

This is a high-severity bug because:
- It affects a core, documented feature
- It crashes on valid user code
- It's not a rare edge case - state management is a standard scanning pattern
- There's no workaround (you can't change scanner state)

## Fix

Implement the missing `begin()` method in the Scanner class. Based on the Scanner's internal structure (it has a `state_name` attribute), the fix is likely:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -XXX,X +XXX,X @@ cdef class Scanner:
+    def begin(self, state_name):
+        """Change the scanner to the specified state."""
+        self.state_name = state_name
```

Additionally, implement the `produce()` method that's also documented but missing.