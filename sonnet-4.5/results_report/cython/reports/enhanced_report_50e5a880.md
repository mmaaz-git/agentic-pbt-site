# Bug Report: Cython.Plex.Scanner Missing Documented Methods

**Target**: `Cython.Plex.Scanners.Scanner`
**Severity**: Invalid
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The Scanner class docstring documents two methods (begin() and produce()) that are not accessible from Python due to being compiled as C-only methods, creating a documentation inconsistency.

## Property-Based Test

```python
import io
from hypothesis import given, strategies as st, settings
from Cython.Plex import *

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=100)
def test_scanner_begin_method_exists(pattern):
    """Test that Scanner has the begin() method as documented in its docstring."""
    lexicon = Lexicon([(Str(pattern), TEXT)])
    scanner = Scanner(lexicon, io.StringIO(pattern))
    assert hasattr(scanner, 'begin'), "Scanner docstring claims begin() method exists, but it doesn't"

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=100)
def test_scanner_produce_method_exists(pattern):
    """Test that Scanner has the produce() method as documented in its docstring."""
    lexicon = Lexicon([(Str(pattern), TEXT)])
    scanner = Scanner(lexicon, io.StringIO(pattern))
    assert hasattr(scanner, 'produce'), "Scanner docstring claims produce() method exists, but it doesn't"

@given(st.text(alphabet='abc', min_size=1, max_size=3))
@settings(max_examples=100)
def test_begin_action_works(pattern):
    """Test that Begin action can perform its documented function."""
    lexicon = Lexicon([(Str(pattern), TEXT)])
    scanner = Scanner(lexicon, io.StringIO(pattern))
    begin_action = Begin('new_state')
    # Begin.perform() should call scanner.begin() according to its implementation
    assert hasattr(begin_action, 'perform'), "Begin action should have perform() method"
    begin_action.perform(scanner, 'test_text')  # This should work without error

# Run the tests
if __name__ == "__main__":
    print("Testing Scanner.begin() method existence...")
    try:
        test_scanner_begin_method_exists()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        print(f"  Failing input: pattern='a'")

    print("\nTesting Scanner.produce() method existence...")
    try:
        test_scanner_produce_method_exists()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        print(f"  Failing input: pattern='a'")

    print("\nTesting Begin action functionality...")
    try:
        test_begin_action_works()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
        print(f"  Failing input: pattern='a'")
    except AttributeError as e:
        print(f"  FAILED with AttributeError: {e}")
        print(f"  Failing input: pattern='a'")
```

<details>

<summary>
**Failing input**: `pattern='a'`
</summary>
```
Testing Scanner.begin() method existence...
  FAILED: Scanner docstring claims begin() method exists, but it doesn't
  Failing input: pattern='a'

Testing Scanner.produce() method existence...
  FAILED: Scanner docstring claims produce() method exists, but it doesn't
  Failing input: pattern='a'

Testing Begin action functionality...
  FAILED: Begin action should have perform() method
  Failing input: pattern='a'
```
</details>

## Reproducing the Bug

```python
import io
from Cython.Plex import *

# Create a simple lexicon with one pattern
lexicon = Lexicon([(Str('hello'), TEXT)])

# Create a scanner with test input
scanner = Scanner(lexicon, io.StringIO('hello'))

# Show Scanner's docstring which documents begin() and produce() methods
print("Scanner class docstring states:")
print('  "begin(state_name) - Causes scanner to change state."')
print('  "produce(value [, text]) - Causes return of a token value..."')

# Test 1: Check if begin() method exists (as documented)
print("\nChecking if Scanner.begin() exists:")
print(f"  hasattr(scanner, 'begin'): {hasattr(scanner, 'begin')}")
if not hasattr(scanner, 'begin'):
    print("  ERROR: Scanner.begin() is documented but not accessible from Python!")

# Test 2: Check if produce() method exists (as documented)
print("\nChecking if Scanner.produce() exists:")
print(f"  hasattr(scanner, 'produce'): {hasattr(scanner, 'produce')}")
if not hasattr(scanner, 'produce'):
    print("  ERROR: Scanner.produce() is documented but not accessible from Python!")

# Test 3: Test Begin action which depends on scanner.begin()
print("\nTesting Begin action (from Actions.py line 82):")
print("  Begin.perform() implementation calls: token_stream.begin(self.state_name)")
begin_action = Begin('new_state')
print(f"  hasattr(begin_action, 'perform'): {hasattr(begin_action, 'perform')}")
if not hasattr(begin_action, 'perform'):
    print("  ERROR: Begin.perform() is not accessible from Python!")
    print("  This means Begin actions cannot be used in lexical specifications")

# Show actual methods available
print("\nActual public methods available on Scanner:")
methods = [m for m in dir(scanner) if not m.startswith('_') and callable(getattr(scanner, m, None))]
for method in sorted(methods):
    print(f"  - {method}")

print("\nConclusion:")
print("  The Scanner class docstring promises begin() and produce() methods")
print("  but they are declared as 'cdef inline' in Scanners.pxd (lines 45-46)")
print("  making them C-only and inaccessible from Python.")
print("  This breaks the Begin action and violates the documented API contract.")
```

<details>

<summary>
AttributeError when trying to use documented methods
</summary>
```
Scanner class docstring states:
  "begin(state_name) - Causes scanner to change state."
  "produce(value [, text]) - Causes return of a token value..."

Checking if Scanner.begin() exists:
  hasattr(scanner, 'begin'): False
  ERROR: Scanner.begin() is documented but not accessible from Python!

Checking if Scanner.produce() exists:
  hasattr(scanner, 'produce'): False
  ERROR: Scanner.produce() is documented but not accessible from Python!

Testing Begin action (from Actions.py line 82):
  Begin.perform() implementation calls: token_stream.begin(self.state_name)
  hasattr(begin_action, 'perform'): False
  ERROR: Begin.perform() is not accessible from Python!
  This means Begin actions cannot be used in lexical specifications

Actual public methods available on Scanner:
  - eof
  - get_position
  - position
  - read

Conclusion:
  The Scanner class docstring promises begin() and produce() methods
  but they are declared as 'cdef inline' in Scanners.pxd (lines 45-46)
  making them C-only and inaccessible from Python.
  This breaks the Begin action and violates the documented API contract.
```
</details>

## Why This Is A Bug

This is classified as **Invalid** rather than a bug because:

1. **Internal Implementation Detail**: The Cython.Plex module is part of Cython's internal implementation infrastructure for its own parser/lexer. The methods exist and work correctly within Cython's compiled C code, but were intentionally made C-only (`cdef inline`) for performance optimization.

2. **Not Intended for Direct Python Use**: The Plex module is primarily used internally by Cython itself. While it has a Python-facing docstring, the actual usage is through Cython's compilation process where these methods are accessible at the C level.

3. **Documentation Artifact**: The docstring appears to be legacy documentation from before full Cythonization, or documentation intended for Cython module developers who work at the C level. This is a documentation inconsistency rather than a functional bug.

4. **Working as Designed**: The methods do exist and function correctly in their intended context (within compiled Cython code). The `cdef inline` declarations in Scanners.pxd (lines 45-46) and Actions.pxd are intentional performance optimizations.

## Relevant Context

The investigation revealed:
- Scanner.begin() and Scanner.produce() are implemented in Scanners.py (lines 331-350)
- They are declared as `cdef inline` in Scanners.pxd (lines 45-46), making them C-only
- Begin.perform() in Actions.py (line 82) calls scanner.begin() internally
- Begin.perform() is also declared as `cdef` in Actions.pxd (line 18)
- These methods work correctly when called from within compiled Cython code
- The Scanner class only exposes 4 methods to Python: eof(), get_position(), position(), and read()

The Plex module is part of Cython's internal infrastructure and these optimizations are likely intentional for the performance-critical task of parsing/lexing during Cython compilation.

## Proposed Fix

Since this is an internal Cython module with intentional performance optimizations, the appropriate fix would be to update the documentation to clarify the module's internal nature:

```diff
--- a/Cython/Plex/Scanners.py
+++ b/Cython/Plex/Scanners.py
@@ -37,12 +37,15 @@ class Scanner:
         Returns the position of the last token read using the
         read() method.

-      begin(state_name)
-        Causes scanner to change state.
-
-      produce(value [, text])
-        Causes return of a token value to the caller of the
-        Scanner.
+      Internal methods (C-level only, not accessible from Python):
+        begin(state_name)
+          Causes scanner to change state.
+          Note: This method is only available within compiled Cython code.
+
+        produce(value [, text])
+          Causes return of a token value to the caller of the Scanner.
+          Note: This method is only available within compiled Cython code.

     """
```

Alternatively, leave the implementation as-is since this is working as designed for Cython's internal use.