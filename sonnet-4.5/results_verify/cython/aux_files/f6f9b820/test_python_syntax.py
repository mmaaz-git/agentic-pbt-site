#!/usr/bin/env python3

# Test 1: Try duplicate else in Python
try:
    code = """
if False:
    print("A")
else:
    print("B")
else:
    print("C")
"""
    compile(code, '<string>', 'exec')
    print("Python accepts duplicate else (unexpected!)")
except SyntaxError as e:
    print(f"Python rejects duplicate else: {e}")

# Test 2: Try elif after else in Python
try:
    code = """
if False:
    print("A")
else:
    print("B")
elif True:
    print("C")
"""
    compile(code, '<string>', 'exec')
    print("Python accepts elif after else (unexpected!)")
except SyntaxError as e:
    print(f"Python rejects elif after else: {e}")