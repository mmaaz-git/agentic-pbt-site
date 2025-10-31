#!/usr/bin/env python3
"""Test to check if Scanner is using a compiled extension module"""

import io
import sys
import inspect
from Cython.Plex import Scanner, Lexicon, Str

# Check the module and class types
print("Scanner module:", Scanner.__module__)
print("Scanner class type:", type(Scanner))
print("Scanner file:", getattr(Scanner, '__file__', 'N/A'))

# Try to get the source code
try:
    source = inspect.getsource(Scanner)
    print("Scanner appears to be pure Python")
except (OSError, TypeError) as e:
    print(f"Cannot get source for Scanner - likely compiled: {e}")

# Check if Scanner is defined in a compiled extension
scanner_module = sys.modules[Scanner.__module__]
print("\nScanner module file:", scanner_module.__file__)
print("Module type:", type(scanner_module))

# Create an instance and check it
lexicon = Lexicon([(Str('x'), 'X_TOKEN')])
scanner_instance = Scanner(lexicon, io.StringIO('x'))
print("\nScanner instance type:", type(scanner_instance))
print("Instance class:", scanner_instance.__class__)

# Check what methods exist on the instance
instance_attrs = [a for a in dir(scanner_instance) if not a.startswith('_')]
print("\nInstance attributes:", sorted(instance_attrs))

# Check if the Python source defines begin but the instance doesn't have it
print("\nChecking method presence:")
print(f"  'begin' in Scanner source file: Will check...")
print(f"  hasattr(Scanner, 'begin'): {hasattr(Scanner, 'begin')}")
print(f"  hasattr(scanner_instance, 'begin'): {hasattr(scanner_instance, 'begin')}")

# Check if this is a Cython extension
if scanner_module.__file__.endswith('.so') or scanner_module.__file__.endswith('.pyd'):
    print("\nThis appears to be a compiled Cython extension!")
    print("The Python source file may not match the compiled version.")
else:
    print("\nThis appears to be pure Python.")