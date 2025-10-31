import numpy.char as char
import numpy as np

# Check docstrings for the functions
functions = ['upper', 'lower', 'capitalize', 'title', 'swapcase', 'strip', 'lstrip', 'rstrip', 'encode', 'decode']

for func_name in functions:
    if hasattr(char, func_name):
        func = getattr(char, func_name)
        print(f"\n{'='*60}")
        print(f"Function: numpy.char.{func_name}")
        print(f"{'='*60}")
        if func.__doc__:
            # Print first few lines of docstring
            lines = func.__doc__.split('\n')
            for i, line in enumerate(lines[:15]):  # Print first 15 lines
                print(line)
            if len(lines) > 15:
                print("... (truncated)")

# Also check the module docstring
print(f"\n{'='*60}")
print(f"numpy.char module docstring:")
print(f"{'='*60}")
if char.__doc__:
    lines = char.__doc__.split('\n')
    for i, line in enumerate(lines[:30]):
        print(line)