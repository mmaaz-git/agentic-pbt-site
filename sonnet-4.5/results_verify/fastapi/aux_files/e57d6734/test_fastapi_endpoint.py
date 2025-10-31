#!/usr/bin/env python3
"""Test the bug with a real FastAPI endpoint"""

from fastapi import FastAPI
import traceback

app = FastAPI()

print("Testing FastAPI endpoint with keyword annotation...")

try:
    # This should crash during definition
    @app.get("/test")
    def test_endpoint(x: "if"):
        return {"x": x}

    print("ERROR: Endpoint was created successfully - this shouldn't happen!")

except SyntaxError as e:
    print(f"Got SyntaxError as expected: {e}")
    traceback.print_exc()

except Exception as e:
    print(f"Got unexpected exception: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test that Python itself allows this in a normal function
print("\nTesting normal Python function with keyword annotation...")
try:
    def normal_func(x: "if"):
        return x

    print("SUCCESS: Normal Python function with keyword annotation works fine")
    print(f"Function annotations: {normal_func.__annotations__}")

except Exception as e:
    print(f"ERROR: Normal Python function failed: {e}")

# Test eval mode vs compile mode
print("\nTesting compile modes...")
print("Trying compile('if', '<string>', 'eval'):")
try:
    code = compile('if', '<string>', 'eval')
    print(f"  ERROR: Compiled successfully as eval (shouldn't work)")
except SyntaxError as e:
    print(f"  SUCCESS: Got SyntaxError as expected: {e}")

print("\nTrying compile('if', '<string>', 'exec'):")
try:
    code = compile('if', '<string>', 'exec')
    print(f"  ERROR: Compiled successfully as exec (shouldn't work)")
except SyntaxError as e:
    print(f"  SUCCESS: Got SyntaxError as expected: {e}")

# Test that valid expressions work
print("\nTesting valid expression annotations...")
try:
    @app.get("/test2")
    def test_endpoint2(x: "str"):
        return {"x": x}

    print("SUCCESS: Valid expression 'str' works fine")

except Exception as e:
    print(f"ERROR: Valid expression failed: {e}")