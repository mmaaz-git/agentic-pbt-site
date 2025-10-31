# Bug Report: Cython.Plex.Scanner Missing Documented Methods

**Target**: `Cython.Plex.Scanners.Scanner`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Scanner` class docstring documents two methods (`begin()` and `produce()`) that do not exist in the actual implementation, violating the API contract.

## Property-Based Test

```python
import io
from hypothesis import given, strategies as st, settings
from Cython.Plex import *

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=100)
def test_scanner_begin_method_exists(pattern):
    lexicon = Lexicon([(Str(pattern), TEXT)])
    scanner = Scanner(lexicon, io.StringIO(pattern))

    assert hasattr(scanner, 'begin'), "Scanner docstring claims begin() method exists, but it doesn't"

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=100)
def test_scanner_produce_method_exists(pattern):
    lexicon = Lexicon([(Str(pattern), TEXT)])
    scanner = Scanner(lexicon, io.StringIO(pattern))

    assert hasattr(scanner, 'produce'), "Scanner docstring claims produce() method exists, but it doesn't"
```

**Failing input**: `pattern='a'` (or any other generated value)

## Reproducing the Bug

```python
import io
from Cython.Plex import *

lexicon = Lexicon([(Str('hello'), TEXT)])
scanner = Scanner(lexicon, io.StringIO('hello'))

print("Attempting to call begin():")
try:
    scanner.begin('some_state')
except AttributeError as e:
    print(f"AttributeError: {e}")

print("\nAttempting to call produce():")
try:
    scanner.produce('TOKEN')
except AttributeError as e:
    print(f"AttributeError: {e}")

print("\nActual methods available:")
methods = [m for m in dir(scanner) if not m.startswith('_') and callable(getattr(scanner, m, None))]
print(f"{methods}")
```

Output:
```
Attempting to call begin():
AttributeError: 'Cython.Plex.Scanners.Scanner' object has no attribute 'begin'

Attempting to call produce():
AttributeError: 'Cython.Plex.Scanners.Scanner' object has no attribute 'produce'

Actual methods available:
['eof', 'get_position', 'position', 'read']
```

## Why This Is A Bug

The Scanner class docstring explicitly documents these methods:

```python
"""
Methods:
  ...
  begin(state_name)
    Causes scanner to change state.

  produce(value [, text])
    Causes return of a token value to the caller of the
    Scanner.
"""
```

However, inspecting the actual Scanner class reveals that neither method exists:
- Available public methods: `eof()`, `get_position()`, `position()`, `read()`
- Missing methods: `begin()`, `produce()`

Users following the documentation will encounter `AttributeError` when trying to use these documented methods. The functionality appears to exist through actions (`Begin` class for state changes), but the documented instance methods do not exist.

## Fix

**Option 1**: Remove the misleading documentation and clarify how to use Begin action instead:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -XX,XX +XX,XX @@ class Scanner:
       position() --> (name, line, col)
         Returns the position of the last token read using the
         read() method.
-
-      begin(state_name)
-        Causes scanner to change state.
-
-      produce(value [, text])
-        Causes return of a token value to the caller of the
-        Scanner.
```

**Option 2**: Implement the missing methods to match the documentation:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -XX,XX +XX,XX @@ class Scanner:
+    def begin(self, state_name):
+        """Causes scanner to change state."""
+        self.state_name = state_name
+
+    def produce(self, value, text=None):
+        """Causes return of a token value to the caller of the Scanner."""
+        if text is None:
+            text = ''
+        return (value, text)
```

Option 1 is simpler and maintains backward compatibility, while Option 2 provides the documented API.