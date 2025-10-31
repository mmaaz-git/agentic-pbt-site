#!/usr/bin/env python3
"""Trace where exactly the encoding error happens."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

def dummy_app(scope, receive, send):
    pass

# Create middleware with non-latin-1 character
middleware = CORSMiddleware(dummy_app, allow_headers=['Ā'], allow_origins=["*"])

print("Inspecting middleware state after initialization:")
print(f"allow_headers: {middleware.allow_headers}")
print(f"preflight_headers: {middleware.preflight_headers}")

# Check if the problematic header is already in preflight_headers
if "Access-Control-Allow-Headers" in middleware.preflight_headers:
    header_value = middleware.preflight_headers["Access-Control-Allow-Headers"]
    print(f"\nAccess-Control-Allow-Headers value: {header_value}")
    print(f"Type: {type(header_value)}")
    print(f"Characters: {[char for char in header_value]}")

    # Try to encode it
    try:
        encoded = header_value.encode('latin-1')
        print("Successfully encoded to latin-1")
    except UnicodeEncodeError as e:
        print(f"Cannot encode to latin-1: {e}")