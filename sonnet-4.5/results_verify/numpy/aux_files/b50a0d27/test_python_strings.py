#!/usr/bin/env python3
# Test Python's string methods with null bytes

print("Testing Python string methods with null bytes:")
print("="*50)

# Test find()
print("str.find() tests:")
print(f"''.find('\\x00') = {('').find('\x00')}")  # Should be -1
print(f"'a\\x00b'.find('\\x00') = {('a\x00b').find('\x00')}")  # Should be 1
print(f"'hello'.find('\\x00') = {('hello').find('\x00')}")  # Should be -1
print()

# Test rfind()
print("str.rfind() tests:")
print(f"''.rfind('\\x00') = {('').rfind('\x00')}")  # Should be -1
print(f"'a\\x00b'.rfind('\\x00') = {('a\x00b').rfind('\x00')}")  # Should be 1
print(f"'hello'.rfind('\\x00') = {('hello').rfind('\x00')}")  # Should be -1
print()

# Test count()
print("str.count() tests:")
print(f"''.count('\\x00') = {('').count('\x00')}")  # Should be 0
print(f"'a\\x00b'.count('\\x00') = {('a\x00b').count('\x00')}")  # Should be 1
print(f"'hello'.count('\\x00') = {('hello').count('\x00')}")  # Should be 0
print()

# Test startswith()
print("str.startswith() tests:")
print(f"''.startswith('\\x00') = {('').startswith('\x00')}")  # Should be False
print(f"'\\x00abc'.startswith('\\x00') = {('\x00abc').startswith('\x00')}")  # Should be True
print(f"'hello'.startswith('\\x00') = {('hello').startswith('\x00')}")  # Should be False
print()

# Test endswith()
print("str.endswith() tests:")
print(f"''.endswith('\\x00') = {('').endswith('\x00')}")  # Should be False
print(f"'abc\\x00'.endswith('\\x00') = {('abc\x00').endswith('\x00')}")  # Should be True
print(f"'hello'.endswith('\\x00') = {('hello').endswith('\x00')}")  # Should be False
print()

# Test index()
print("str.index() tests:")
try:
    result = ('').index('\x00')
    print(f"''.index('\\x00') = {result}")
except ValueError as e:
    print(f"''.index('\\x00') raised ValueError: {e}")

try:
    result = ('a\x00b').index('\x00')
    print(f"'a\\x00b'.index('\\x00') = {result}")
except ValueError as e:
    print(f"'a\\x00b'.index('\\x00') raised ValueError: {e}")

try:
    result = ('hello').index('\x00')
    print(f"'hello'.index('\\x00') = {result}")
except ValueError as e:
    print(f"'hello'.index('\\x00') raised ValueError: {e}")

print()
print("Python treats null bytes (\\x00) as regular characters in strings.")