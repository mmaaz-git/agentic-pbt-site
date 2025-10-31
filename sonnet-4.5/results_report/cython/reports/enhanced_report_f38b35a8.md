# Bug Report: Cython.Plex.Regexps.RawCodeRange.calc_str AttributeError on String Conversion

**Target**: `Cython.Plex.Regexps.RawCodeRange.calc_str`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `RawCodeRange.calc_str` method crashes with an AttributeError when converting a RawCodeRange object to string, because it references non-existent attributes `self.code1` and `self.code2` instead of using the actual `self.range` tuple.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pytest
from Cython.Plex.Regexps import RawCodeRange

@given(st.integers(min_value=0, max_value=200),
       st.integers(min_value=0, max_value=200))
@settings(max_examples=300)
def test_rawcoderange_str_method(code1, code2):
    if code1 >= code2:
        return

    rcr = RawCodeRange(code1, code2)

    try:
        str_repr = str(rcr)
        assert str_repr is not None
    except AttributeError as e:
        if 'code1' in str(e) or 'code2' in str(e):
            pytest.fail(f"RawCodeRange.calc_str references non-existent attributes: {e}")

if __name__ == "__main__":
    test_rawcoderange_str_method()
```

<details>

<summary>
**Failing input**: `code1=0, code2=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 15, in test_rawcoderange_str_method
    str_repr = str(rcr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py", line 149, in __str__
    return self.calc_str()
           ~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py", line 224, in calc_str
    return "CodeRange(%d,%d)" % (self.code1, self.code2)
                                 ^^^^^^^^^^
AttributeError: 'RawCodeRange' object has no attribute 'code1'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 22, in <module>
    test_rawcoderange_str_method()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 6, in test_rawcoderange_str_method
    st.integers(min_value=0, max_value=200))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 19, in test_rawcoderange_str_method
    pytest.fail(f"RawCodeRange.calc_str references non-existent attributes: {e}")
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: RawCodeRange.calc_str references non-existent attributes: 'RawCodeRange' object has no attribute 'code1'
Falsifying example: test_rawcoderange_str_method(
    code1=0,
    code2=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/44/hypo.py:12
```
</details>

## Reproducing the Bug

```python
from Cython.Plex.Regexps import RawCodeRange

# Create a RawCodeRange instance with valid code range
rcr = RawCodeRange(50, 60)

# Try to get string representation
try:
    s = str(rcr)
    print(f"String representation: {s}")
except AttributeError as e:
    print(f"AttributeError: {e}")
    print(f"rcr.range exists: {hasattr(rcr, 'range')}")
    print(f"rcr.range value: {rcr.range if hasattr(rcr, 'range') else 'N/A'}")
    print(f"rcr.code1 exists: {hasattr(rcr, 'code1')}")
    print(f"rcr.code2 exists: {hasattr(rcr, 'code2')}")
```

<details>

<summary>
AttributeError when converting RawCodeRange to string
</summary>
```
AttributeError: 'RawCodeRange' object has no attribute 'code1'
rcr.range exists: True
rcr.range value: (50, 60)
rcr.code1 exists: False
rcr.code2 exists: False
```
</details>

## Why This Is A Bug

The RawCodeRange class has a clear implementation mismatch between its `__init__` method and `calc_str` method. The `__init__` method (lines 208-211 in Regexps.py) stores the code range as a tuple in `self.range = (code1, code2)`, which is consistent with the class attribute documentation on line 204 that states `range = None  # (code, code)`. However, the `calc_str` method on line 224 attempts to access `self.code1` and `self.code2`, which are never defined as instance attributes.

This violates the expected Python behavior where all objects should have a working string representation. The base RE class's `__str__` method (lines 145-149) calls `calc_str()` to generate the string representation, which causes the crash whenever `str()` is called on a RawCodeRange object. Even though the class is documented as "For internal use only," internal code still needs string representation for debugging, logging, and error messages.

## Relevant Context

The RawCodeRange class is part of Cython's Plex module, which handles regular expression parsing and lexical analysis. The class inherits from the RE base class which provides a standard pattern for string representation through the `calc_str()` method. Each RE subclass is expected to implement its own `calc_str()` method correctly.

Key code locations:
- RawCodeRange class definition: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py:196-225`
- Base RE class `__str__` method: `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py:145-149`
- Class attribute documentation clearly shows `range` stores a tuple: line 204

The bug doesn't affect the core functionality of the RawCodeRange class (pattern matching still works), but it breaks debugging and logging capabilities whenever the object needs to be converted to a string.

## Proposed Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -221,7 +221,7 @@ class RawCodeRange(RE):
                 initial_state.add_transition(self.lowercase_range, final_state)

     def calc_str(self):
-        return "CodeRange(%d,%d)" % (self.code1, self.code2)
+        return "CodeRange(%d,%d)" % self.range
```