#!/usr/bin/env python3
"""Test if FastAPI would actually work with path parameters containing newlines"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import re

app = FastAPI()

# Try to define a route with a newline in the parameter name
# This is the critical question: would this ever work?

# Test 1: Can we even define a Python function with newline in parameter?
try:
    code = """
def test_func(param_with
newline):
    return param_with
newline
"""
    exec(code)
    print("Function with newline in param name: SUCCESS (unexpected!)")
except SyntaxError as e:
    print(f"Function with newline in param name: FAILED - {e}")

# Test 2: Can we use exec to create such a function?
try:
    # This won't work because Python identifiers can't have newlines
    exec("""
def test_func(**kwargs):
    return kwargs.get('param\nname')
""")
    print("Function with newline via kwargs: SUCCESS")
except Exception as e:
    print(f"Function with newline via kwargs: FAILED - {e}")

# Test 3: Would FastAPI route even accept such a path?
# Let's test with actual FastAPI
print("\n=== Testing FastAPI Routes ===")

# Normal path parameter
@app.get("/items/{item_id}")
async def read_item(item_id: str):
    return {"item_id": item_id}

# Path with space (questionable but let's try)
try:
    @app.get("/space/{item id}")  # This won't match Python param name
    async def read_space(item_id: str):  # Can't have space in Python param
        return {"item_id": item_id}
    print("Route with space in param: CREATED")
except Exception as e:
    print(f"Route with space in param: FAILED - {e}")

# Let's test what paths FastAPI actually accepts
client = TestClient(app)

# Test normal path
response = client.get("/items/123")
print(f"\nNormal path '/items/123': {response.status_code} - {response.json()}")

# Let's look at what the regex actually extracts
test_paths = [
    "/{valid_param}",
    "/{ space }",
    "/{\ttab\t}",
    "/{\nnewline\n}",
    "/{123numbers}",
    "/{-dash-}",
    "/{.dot.}",
]

print("\n=== Regex Extraction Test ===")
for path in test_paths:
    params = set(re.findall("{(.*?)}", path))
    params_fixed = set(re.findall("{([^}]*)}", path))
    print(f"Path: {repr(path):25} | Current: {str(params):20} | Fixed: {str(params_fixed):20}")

# The real question: Even if we extract these names, could they ever map to Python function parameters?
print("\n=== Python Identifier Validity ===")
test_names = ['valid_param', ' space ', '\ttab\t', '\nnewline\n', '123numbers', '-dash-', '.dot.']
for name in test_names:
    try:
        # Test if it's a valid Python identifier
        exec(f"{name} = 1")
        print(f"{repr(name):15} - VALID Python identifier")
    except:
        print(f"{repr(name):15} - INVALID Python identifier")