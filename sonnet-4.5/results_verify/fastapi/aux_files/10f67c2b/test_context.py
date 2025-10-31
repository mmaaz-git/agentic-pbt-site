#!/usr/bin/env python3
"""Test to understand how is_body_allowed_for_status_code is used in context"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

app = FastAPI()

# Test 1: Normal usage with integer status codes
@app.get("/normal", status_code=200)
def normal():
    return {"message": "ok"}

# Test 2: Can we use pattern strings?
@app.get("/pattern", status_code="2XX")
def pattern():
    return {"message": "pattern"}

# Test 3: What about responses dict with pattern strings?
@app.get("/responses", responses={"200": {"description": "Success"}, "4XX": {"description": "Client error"}})
def responses():
    return {"message": "responses"}

# Test 4: What happens if we try to use an invalid string as status_code?
try:
    @app.get("/invalid", status_code="invalid")
    def invalid():
        return {"message": "invalid"}
    print("✗ Route with status_code='invalid' created without error!")
except Exception as e:
    print(f"✓ Route with status_code='invalid' failed as expected: {e}")

# Test 5: What about numeric strings?
@app.get("/numeric", status_code="404")
def numeric():
    return {"message": "numeric"}

print("\nRoutes successfully created:")
for route in app.routes:
    if hasattr(route, 'path'):
        print(f"  {route.path}")

# Now let's test the openapi generation which might use is_body_allowed_for_status_code
print("\nGenerating OpenAPI schema...")
try:
    schema = app.openapi()
    print("✓ OpenAPI schema generated successfully")

    # Check what status codes are in the schema
    if 'paths' in schema:
        for path, methods in schema['paths'].items():
            for method, details in methods.items():
                if 'responses' in details:
                    print(f"\n  {path} ({method}):")
                    for status_code in details['responses'].keys():
                        print(f"    - {status_code}")
except Exception as e:
    print(f"✗ OpenAPI generation failed: {e}")