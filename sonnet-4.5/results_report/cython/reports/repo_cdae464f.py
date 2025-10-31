"""Minimal reproduction of the Cython.Distutils.read_setup_file bug."""

import tempfile
import os
from Cython.Distutils.extension import read_setup_file

# Create a temporary Setup file with a -D macro definition
with tempfile.NamedTemporaryFile(mode='w', suffix='.setup', delete=False) as f:
    f.write("testmod test.c -DFOO=bar")
    temp_path = f.name

try:
    # Read the setup file
    extensions = read_setup_file(temp_path)

    # Get the first (and only) extension
    ext = extensions[0]

    # Check the define_macros
    print(f"Extension name: {ext.name}")
    print(f"Extension sources: {ext.sources}")
    print(f"Number of macros defined: {len(ext.define_macros)}")

    if ext.define_macros:
        name, value = ext.define_macros[0]
        print(f"\nExpected macro: ('FOO', 'bar')")
        print(f"Actual macro:   ('{name}', '{value}')")
        print(f"\nBUG: The value 'bar' was truncated to '{value}' - first character is missing!")
finally:
    # Clean up the temporary file
    os.unlink(temp_path)