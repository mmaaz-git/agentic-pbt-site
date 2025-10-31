#!/usr/bin/env python3
"""Test the impact of empty parameter names in FastAPI"""

from fastapi import FastAPI, Path
from fastapi.testclient import TestClient

app = FastAPI()

# Test 1: Normal route with proper parameters
@app.get("/users/{user_id}/posts/{post_id}")
def get_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}

# Test 2: Route with empty parameter (will this even work?)
# This should fail since we can't have an empty function parameter
try:
    exec("""
@app.get("/items/{}/details")
def get_item_details():  # Can't have empty param name
    return {"msg": "test"}
""")
    print("Empty parameter route created (unexpected)")
except Exception as e:
    print(f"Empty parameter route failed: {type(e).__name__}")

# Test 3: Check if path matching works with empty braces
# Let's see if FastAPI can handle paths with empty braces
client = TestClient(app)

print("\n=== Testing normal route ===")
response = client.get("/users/123/posts/456")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

print("\n=== Testing OpenAPI schema generation ===")
try:
    schema = app.openapi()
    print(f"OpenAPI schema generated: {bool(schema)}")

    # Check paths in schema
    if "paths" in schema:
        for path in schema["paths"]:
            print(f"  Path: {path}")
            if "parameters" in schema["paths"][path].get("get", {}):
                params = schema["paths"][path]["get"]["parameters"]
                print(f"    Parameters: {[p.get('name') for p in params]}")

except Exception as e:
    print(f"Schema generation error: {e}")

print("\n=== Direct test of get_path_param_names ===")
from fastapi.utils import get_path_param_names

test_paths = [
    "/users/{user_id}",
    "/items/{}",
    "/{}/{}",
    "/mixed/{id}/{}/end"
]

for path in test_paths:
    params = get_path_param_names(path)
    print(f"Path: {path:30} -> Params: {params}")
    if '' in params:
        print(f"  ⚠️  Contains empty parameter name!")

print("\n=== Test path parameter matching ===")
# What happens when we try to match function params to path params?
def simulate_param_matching(path: str, func_params: list):
    """Simulate how FastAPI matches function params to path params"""
    path_params = get_path_param_names(path)
    print(f"Path: {path}")
    print(f"  Path params: {path_params}")
    print(f"  Function params: {func_params}")

    for param in func_params:
        if param in path_params:
            print(f"    ✓ {param} is a path parameter")
        else:
            print(f"    ✗ {param} is NOT a path parameter")

    # Check for unmatched path params
    unmatched = path_params - set(func_params)
    if unmatched:
        print(f"  ⚠️  Unmatched path params: {unmatched}")

    print()

simulate_param_matching("/users/{user_id}", ["user_id", "limit"])
simulate_param_matching("/items/{}", ["item_id"])
simulate_param_matching("/things/{id}/{}", ["id", "name"])