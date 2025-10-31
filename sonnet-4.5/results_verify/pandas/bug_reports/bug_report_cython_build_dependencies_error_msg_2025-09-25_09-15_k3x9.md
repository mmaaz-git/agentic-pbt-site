# Bug Report: Cython.Build.Dependencies.DependencyTree.included_files Swapped Error Arguments

**Target**: `Cython.Build.Dependencies.DependencyTree.included_files` at line 544
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error message for unable to locate included files has its arguments swapped, printing "Unable to locate 'X' referenced from 'Y'" when it should say "Unable to locate 'Y' referenced from 'X'".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock

@settings(max_examples=100)
@given(
    filename=st.text(min_size=1, max_size=20, alphabet='abcdefgh/_.'),
    include=st.text(min_size=1, max_size=20, alphabet='abcdefgh/_.pyx')
)
def test_included_files_error_message_format(filename, include):
    from Cython.Build.Dependencies import DependencyTree
    import io
    import sys

    mock_context = Mock()
    mock_context.find_include_file = Mock(return_value=None)

    tree = DependencyTree(mock_context, quiet=False)

    old_parse = tree.parse_dependencies
    tree.parse_dependencies = Mock(return_value=([], [include], [], None))

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        tree.included_files(filename)
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    if include in output and filename in output:
        assert output.find(include) < output.find(filename), \
            f"Error should say 'Unable to locate {include!r} referenced from {filename!r}'"
```

**Failing input**: Any filename and include name when include file cannot be found

## Reproducing the Bug

```python
filename = "mymodule.pyx"
include = "missing_header.pxd"

error_template = "Unable to locate '%s' referenced from '%s'"
actual_output = error_template % (filename, include)
expected_output = error_template % (include, filename)

print(f"Actual:   {actual_output}")
print(f"Expected: {expected_output}")

assert actual_output == "Unable to locate 'mymodule.pyx' referenced from 'missing_header.pxd'"
assert expected_output == "Unable to locate 'missing_header.pxd' referenced from 'mymodule.pyx'"
print("\nBUG: The arguments are swapped!")
```

Output:
```
Actual:   Unable to locate 'mymodule.pyx' referenced from 'missing_header.pxd'
Expected: Unable to locate 'missing_header.pxd' referenced from 'mymodule.pyx'

BUG: The arguments are swapped!
```

## Why This Is A Bug

At `Cython/Build/Dependencies.py:544`:
```python
print("Unable to locate '%s' referenced from '%s'" % (filename, include))
```

The function is trying to locate the `include` file (referenced from `filename`), but the format arguments are backwards. The message should indicate:
- What we can't find: `include`
- Where it's referenced from: `filename`

But the code produces: "Unable to locate 'filename' referenced from 'include'" which is backwards and confusing for users trying to debug missing include files.

## Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -541,7 +541,7 @@ class DependencyTree:
                 all.add(include_path)
                 all.update(self.included_files(include_path))
             elif not self.quiet:
-                print("Unable to locate '%s' referenced from '%s'" % (filename, include))
+                print("Unable to locate '%s' referenced from '%s'" % (include, filename))
         return all

     @cached_method
```