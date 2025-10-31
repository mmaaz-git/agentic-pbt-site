# Bug Report: Cython.Plex Scanner State Management Methods Not Accessible from Python

**Target**: `Cython.Plex.Scanners.Scanner` and `Cython.Plex.Actions.Begin`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Scanner's `begin()` and `produce()` methods are documented as public but are inaccessible from Python due to incorrect Cython declarations, causing crashes when using the Begin action for state management in lexical scanners.

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

if __name__ == '__main__':
    test_begin_action_changes_state()
```

<details>

<summary>
**Failing input**: `trigger_pattern='a'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 24, in <module>
    test_begin_action_changes_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 6, in test_begin_action_changes_state
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 20, in test_begin_action_changes_state
    token1, text1 = scanner.read()
                    ~~~~~~~~~~~~^^
  File "Cython/Plex/Scanners.py", line 114, in Cython.Plex.Scanners.Scanner.read
  File "Cython/Plex/Scanners.py", line 128, in Cython.Plex.Scanners.Scanner.read
  File "Cython/Plex/Actions.py", line 82, in Cython.Plex.Actions.Begin.perform
AttributeError: 'Cython.Plex.Scanners.Scanner' object has no attribute 'begin'
Falsifying example: test_begin_action_changes_state(
    trigger_pattern='a',  # or any other generated value
)
```
</details>

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

<details>

<summary>
AttributeError on scanner.read()
</summary>
```
AttributeError: 'Cython.Plex.Scanners.Scanner' object has no attribute 'begin'
```
</details>

## Why This Is A Bug

This violates documented behavior and breaks core functionality. The Scanner class documentation (Scanners.py lines 40-45) explicitly lists `begin(state_name)` and `produce(value [, text])` as public methods. The Lexicon documentation (Lexicons.py lines 101-103) describes two ways to change scanner state, including "Calling the begin(state_name) method of the Scanner." The Begin action (Actions.py line 82) relies on calling `scanner.begin()` to function.

The root cause is a mismatch between the Python source implementation and the Cython declaration file. In Scanners.py (lines 331-335 and 337-349), both methods are implemented as regular Python methods using `def`. However, in Scanners.pxd (lines 45-46), they are declared as `cdef inline`, making them C-only methods inaccessible from Python. The compiled .so module follows the .pxd declarations, causing the AttributeError.

This completely breaks multi-state lexical analysis, a fundamental feature for parsing complex languages with different contexts (strings, comments, etc.). There is no workaround since scanner state cannot be changed through any other means.

## Relevant Context

The Cython.Plex module is used for lexical analysis in Cython's parsing infrastructure. State management is essential for handling different lexical contexts. The mismatch appears unintentional - the Python source correctly implements these as public methods, but the Cython compilation directives hide them.

Key files:
- `/Cython/Plex/Scanners.py` - Contains correct Python implementations
- `/Cython/Plex/Scanners.pxd` - Contains incorrect Cython declarations
- `/Cython/Plex/Actions.py` - Begin action depends on scanner.begin()
- `/Cython/Plex/Lexicons.py` - Documents state management as core feature

## Proposed Fix

Change the Cython declaration file to expose these methods as Python-callable:

```diff
--- a/Cython/Plex/Scanners.pxd
+++ b/Cython/Plex/Scanners.pxd
@@ -42,6 +42,6 @@ cdef class Scanner:

     cdef run_machine_inlined(self)

-    cdef inline begin(self, state)
-    cdef inline produce(self, value, text = *)
+    cpdef begin(self, state)
+    cpdef produce(self, value, text = *)
```