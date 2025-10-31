# Bug Report: Cython.Distutils.read_setup_file Macro Value Truncation

**Target**: `Cython.Distutils.extension.read_setup_file`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `read_setup_file` function incorrectly parses `-D` macro definitions with values, truncating the first character of the macro value due to an off-by-one error.

## Property-Based Test

```python
import tempfile
import os
from hypothesis import given, strategies as st
from Cython.Distutils.extension import read_setup_file


@given(
    st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=48, max_codepoint=122))
    .filter(lambda x: '"' not in x and "'" not in x and ' ' not in x)
)
def test_define_macro_value_parsing(macro_name, macro_value):
    macro_def = f"{macro_name}={macro_value}"
    setup_line = f"testmod test.c -D{macro_def}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.setup', delete=False) as f:
        f.write(setup_line)
        f.flush()
        temp_path = f.name

    try:
        extensions = read_setup_file(temp_path)
        ext = extensions[0]

        assert len(ext.define_macros) == 1
        actual_name, actual_value = ext.define_macros[0]
        assert actual_name == macro_name
        assert actual_value == macro_value
    finally:
        os.unlink(temp_path)
```

**Failing input**: `macro_name='A'`, `macro_value='0'` (among many others)

## Reproducing the Bug

```python
import tempfile
import os
from Cython.Distutils.extension import read_setup_file

with tempfile.NamedTemporaryFile(mode='w', suffix='.setup', delete=False) as f:
    f.write("testmod test.c -DFOO=bar")
    temp_path = f.name

extensions = read_setup_file(temp_path)
name, value = extensions[0].define_macros[0]
print(f"Expected: ('FOO', 'bar')")
print(f"Actual:   ('{name}', '{value}')")
os.unlink(temp_path)
```

Output:
```
Expected: ('FOO', 'bar')
Actual:   ('FOO', 'ar')
```

## Why This Is A Bug

When parsing Setup files, the `-DNAME=value` syntax should set a preprocessor macro `NAME` to `value`. The current implementation incorrectly skips the first character of the value, causing `-DFOO=bar` to define `FOO` as `ar` instead of `bar`, and `-DA=1` to define `A` as an empty string instead of `1`. This breaks the fundamental contract of the Setup file format and would cause incorrect compilation of C extensions.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -66,7 +66,7 @@ def read_setup_file(filename):
                     if equals == -1:  # bare "-DFOO" -- no value
                         ext.define_macros.append((value, None))
                     else:  # "-DFOO=blah"
-                        ext.define_macros.append((value[0:equals], value[equals + 2 :]))
+                        ext.define_macros.append((value[0:equals], value[equals + 1 :]))
                 elif switch == "-U":
                     ext.undef_macros.append(value)
                 elif switch == "-C":  # only here 'cause makesetup has it!
```