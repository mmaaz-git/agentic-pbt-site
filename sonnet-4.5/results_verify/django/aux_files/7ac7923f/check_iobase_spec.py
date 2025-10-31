#!/usr/bin/env python3
"""
Check the io.IOBase specification and contract
"""

import io
import inspect

print("=" * 60)
print("io.IOBase writable() specification")
print("=" * 60)
print()

# Check the IOBase class
print("io.IOBase class documentation:")
print("-" * 40)
print(io.IOBase.__doc__)
print()

print("io.IOBase.writable() method documentation:")
print("-" * 40)
print(io.IOBase.writable.__doc__)
print()

# Check if FileProxyMixin is intended to be compatible with io.IOBase
print("io.IOBase methods that file-like objects should implement:")
print("-" * 40)
for name, method in inspect.getmembers(io.IOBase):
    if not name.startswith('_') and callable(method):
        print(f"  {name}()")

print()
print("Key finding:")
print("-" * 40)
print("The io.IOBase.writable() method is part of the standard Python")
print("file-like object protocol. Django's FileProxyMixin.writable()")
print("should correctly implement this protocol to be compatible with")
print("Python's file I/O system.")
print()
print("The contract states that writable() should return True if the")
print("file was opened for writing, and False otherwise.")
print()
print("FileProxyMixin acts as a proxy, so when the underlying file")
print("doesn't have a writable() method, it must deduce writability")
print("from the mode string. The current implementation incorrectly")
print("only checks for 'w' in the mode, violating the contract for")
print("modes like 'r+', 'a', 'a+', 'x', and 'x+'.")