# Bug Report: Cython.Plex.Scanner Missing Documented Methods

**Target**: `Cython.Plex.Scanners.Scanner`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Scanner` class docstring documents two methods (`begin()` and `produce()`) that do not exist in the actual implementation, violating the API contract and misleading users.

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

**Failing input**: `pattern='a'` (or any other value)

## Reproducing the Bug

```python
import io
from Cython.Plex import *

lexicon = Lexicon([(Str('hello'), TEXT)])
scanner = Scanner(lexicon, io.StringIO('hello'))

try:
    scanner.begin('some_state')
except AttributeError as e:
    print(f'begin() error: {e}')

try:
    scanner.produce('TOKEN')
except AttributeError as e:
    print(f'produce() error: {e}')

methods = [m for m in dir(scanner) if not m.startswith('_') and callable(getattr(scanner, m, None))]
print(f'Actual methods: {methods}')
```

Output:
```
begin() error: 'Cython.Plex.Scanners.Scanner' object has no attribute 'begin'
produce() error: 'Cython.Plex.Scanners.Scanner' object has no attribute 'produce'
Actual methods: ['eof', 'get_position', 'position', 'read']
```

## Why This Is A Bug

The Scanner class docstring explicitly documents these methods:

```
Methods:
  begin(state_name)
    Causes scanner to change state.

  produce(value [, text])
    Causes return of a token value to the caller of the Scanner.
```

However, the actual Scanner implementation only provides:
- `eof()` - check if at end of file
- `get_position()` - get current position
- `position()` - get last token position
- `read()` - read next token

Users following the documentation will encounter `AttributeError` when attempting to use `begin()` or `produce()`. This is particularly problematic because:

1. **State management is broken** - The documented way to change scanner state (`begin()`) doesn't work, and the Begin action actually relies on this method existing (see related bug report).
2. **No workaround** - Without `begin()`, users cannot programmatically change scanner state.
3. **Documentation misleads users** - The docstring is part of the public API contract.

## Fix

**Option 1**: Remove the misleading documentation:

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

**Option 2**: Implement the missing methods (recommended):

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -XXX,X +XXX,X @@ cdef class Scanner:
+    def begin(self, state_name):
+        """Causes scanner to change state."""
+        self.state_name = state_name
+
+    def produce(self, value, text=None):
+        """Causes return of a token value to the caller of the Scanner."""
+        if text is None:
+            text = self.text
+        self.queue.append((value, text))
```

Option 2 is strongly recommended because the Begin action already depends on `begin()` existing.