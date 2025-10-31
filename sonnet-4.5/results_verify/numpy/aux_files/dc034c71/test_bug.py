import numpy as np
import numpy.char as char

print("Testing numpy.char.find with null bytes:")
print("=" * 50)

strings = ['', 'test', 'hello world', 'abc']

for s in strings:
    py_result = s.find('\x00')
    np_result = int(char.find(s, '\x00'))

    print(f"String: {repr(s)}")
    print(f"  Python find: {py_result}")
    print(f"  NumPy find:  {np_result}")
    print(f"  Match: {py_result == np_result}")
    print()

print("Testing with strings that contain null bytes:")
print("=" * 50)

test_strings = ['test\x00ing', '\x00start', 'end\x00', 'mid\x00dle']
for s in test_strings:
    py_result = s.find('\x00')
    np_result = int(char.find(s, '\x00'))

    print(f"String: {repr(s)}")
    print(f"  Python find: {py_result}")
    print(f"  NumPy find:  {np_result}")
    print(f"  Match: {py_result == np_result}")
    print()

print("Testing numpy.char.rfind with null bytes:")
print("=" * 50)

for s in strings:
    py_result = s.rfind('\x00')
    np_result = int(char.rfind(s, '\x00'))

    print(f"String: {repr(s)}")
    print(f"  Python rfind: {py_result}")
    print(f"  NumPy rfind:  {np_result}")
    print(f"  Match: {py_result == np_result}")
    print()