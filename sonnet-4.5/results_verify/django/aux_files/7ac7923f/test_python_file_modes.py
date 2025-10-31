#!/usr/bin/env python3
"""
Test Python's file mode specifications and the io.IOBase.writable() protocol
"""

import io
import tempfile
import os

print("Python File Modes Documentation Test")
print("=" * 60)
print()

# Test Python's built-in documentation
print("Python's io.IOBase.writable() method:")
print("-" * 40)
print(io.IOBase.writable.__doc__)
print()

# Test all standard file modes
print("Testing all Python file modes for writability:")
print("-" * 40)

# Standard Python file modes according to documentation:
# 'r': read (default)
# 'w': write, truncating the file first
# 'x': exclusive creation, failing if the file already exists
# 'a': append, writing to the end of file if it exists
# 'b': binary mode
# 't': text mode (default)
# '+': open for updating (reading and writing)

modes = [
    ('r', 'read only', False),
    ('r+', 'read/write', True),
    ('rb', 'read binary', False),
    ('r+b', 'read/write binary', True),
    ('w', 'write only', True),
    ('w+', 'write/read', True),
    ('wb', 'write binary', True),
    ('w+b', 'write/read binary', True),
    ('a', 'append', True),
    ('a+', 'append/read', True),
    ('ab', 'append binary', True),
    ('a+b', 'append/read binary', True),
    ('x', 'exclusive create', True),
    ('x+', 'exclusive create/read', True),
    ('xb', 'exclusive create binary', True),
    ('x+b', 'exclusive create/read binary', True),
]

print(f"{'Mode':<6} {'Description':<25} {'Expected':<10} {'Actual':<10} {'Match':<10}")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
    tmp_path = tmp.name

try:
    for mode, desc, expected_writable in modes:
        try:
            # For 'x' modes, ensure file doesn't exist
            if 'x' in mode:
                try:
                    os.remove(tmp_path)
                except:
                    pass
            else:
                # Ensure file exists for other modes
                if not os.path.exists(tmp_path):
                    with open(tmp_path, 'w') as f:
                        f.write('test')

            with open(tmp_path, mode) as f:
                actual = f.writable() if hasattr(f, 'writable') else None
                match = "YES" if actual == expected_writable else "NO"
                print(f"{mode:<6} {desc:<25} {expected_writable!s:<10} {actual!s:<10} {match:<10}")

        except Exception as e:
            print(f"{mode:<6} {desc:<25} Error: {e}")

finally:
    try:
        os.remove(tmp_path)
    except:
        pass

print()
print("Python File Mode Rules:")
print("-" * 40)
print("1. 'r' alone: Read-only, NOT writable")
print("2. 'w' in mode: Write mode, IS writable")
print("3. 'a' in mode: Append mode, IS writable")
print("4. 'x' in mode: Exclusive creation, IS writable")
print("5. '+' in mode: Update mode (read+write), IS writable")
print()
print("Therefore, a file is writable if ANY of these are true:")
print("  - 'w' in mode")
print("  - 'a' in mode")
print("  - 'x' in mode")
print("  - '+' in mode")
print()
print("Django's current implementation only checks for 'w', missing 'a', 'x', and '+'!")