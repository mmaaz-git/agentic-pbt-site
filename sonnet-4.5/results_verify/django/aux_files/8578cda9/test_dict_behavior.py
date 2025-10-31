#!/usr/bin/env python3
"""Test how Python's dict handles **kwargs"""

print("Test 1: dict() with kwargs")
d1 = dict(foo="bar", baz=42)
print(f"dict(foo='bar', baz=42) = {d1}")

print("\nTest 2: dict subclass with **kwargs")
class MyDict(dict):
    def __init__(self, *args, **kwargs):
        print(f"  MyDict received args={args}, kwargs={kwargs}")
        super().__init__(*args, **kwargs)

md = MyDict(foo="bar", baz=42)
print(f"MyDict(foo='bar', baz=42) = {md}")

print("\nTest 3: dict subclass with WRONG super call (using *kwargs)")
class BrokenDict(dict):
    def __init__(self, *args, **kwargs):
        print(f"  BrokenDict received args={args}, kwargs={kwargs}")
        # This is the bug - using *kwargs instead of **kwargs
        try:
            super().__init__(*args, *kwargs)
            print("  Success with *kwargs")
        except Exception as e:
            print(f"  Failed with *kwargs: {e}")
            print("  Trying with **kwargs...")
            super().__init__(*args, **kwargs)
            print("  Success with **kwargs")

try:
    bd = BrokenDict(foo="bar")
    print(f"BrokenDict(foo='bar') = {bd}")
except Exception as e:
    print(f"Failed to create BrokenDict: {e}")