# Bug Report: Cython.Distutils.extension.read_setup_file Macro Value Parsing

**Target**: `Cython.Distutils.extension.read_setup_file`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_setup_file` function incorrectly parses `-D` macro definitions with values, dropping the first character of the macro value due to an off-by-one error in string slicing.

## Property-Based Test

```python
import os
import tempfile
from hypothesis import given, strategies as st, settings
from Cython.Distutils.extension import read_setup_file


valid_identifiers = st.from_regex(r'[a-zA-Z_][a-zA-Z0-9_]{0,20}', fullmatch=True)
safe_macro_values = st.text(
    alphabet=st.characters(
        blacklist_categories={'Cs'},
        blacklist_characters='\n\r\'" \t\\',
    ),
    min_size=1,
    max_size=30
)


@given(
    module_name=valid_identifiers,
    macro_name=valid_identifiers,
    macro_value=safe_macro_values
)
@settings(max_examples=200)
def test_define_macro_value_preserved(module_name, macro_name, macro_value):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"{module_name} source.c -D{macro_name}={macro_value}\n")
        temp_file = f.name

    try:
        extensions = read_setup_file(temp_file)
        ext = extensions[0]
        define_macros_dict = dict(ext.define_macros)

        assert macro_name in define_macros_dict
        assert define_macros_dict[macro_name] == macro_value, \
            f"Expected {macro_value!r}, got {define_macros_dict[macro_name]!r}"
    finally:
        os.unlink(temp_file)
```

**Failing input**: `macro_value='0'` (or any single-character value, or longer values where the first character matters)

## Reproducing the Bug

```python
import os
import tempfile
from Cython.Distutils.extension import read_setup_file

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write("mymodule source.c -DVERSION=1.2.3\n")
    temp_file = f.name

try:
    extensions = read_setup_file(temp_file)
    ext = extensions[0]
    define_macros_dict = dict(ext.define_macros)

    print(f"Expected: VERSION=1.2.3")
    print(f"Actual:   VERSION={define_macros_dict['VERSION']}")
finally:
    os.unlink(temp_file)
```

Output:
```
Expected: VERSION=1.2.3
Actual:   VERSION=.2.3
```

## Why This Is A Bug

When parsing `-DFOO=value` syntax in setup files, the code should preserve the entire macro value. Instead, it skips the first character due to incorrect string slicing (`value[equals + 2:]` instead of `value[equals + 1:]`). This affects any use of `-D` flags with values in Cython setup files.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -65,7 +65,7 @@ def read_setup_file(filename):  # noqa: C901
                     if equals == -1:  # bare "-DFOO" -- no value
                         ext.define_macros.append((value, None))
                     else:  # "-DFOO=blah"
-                        ext.define_macros.append((value[0:equals], value[equals + 2 :]))
+                        ext.define_macros.append((value[0:equals], value[equals + 1 :]))
                 elif switch == "-U":
                     ext.undef_macros.append(value)
                 elif switch == "-C":  # only here 'cause makesetup has it!
```