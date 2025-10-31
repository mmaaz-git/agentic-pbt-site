#!/usr/bin/env python3
"""Minimal reproduction of circular dependency bug in FastAPI's get_flat_dependant"""

from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant

# Create two dependants with circular references
dep1 = Dependant(call=lambda: "dep1", name="dep1")
dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])

# Create the circular reference
dep1.dependencies.append(dep2)

# This will cause RecursionError
try:
    flat = get_flat_dependant(dep1)
    print("Success: get_flat_dependant completed")
    print(f"Result type: {type(flat)}")
except RecursionError as e:
    print("RecursionError occurred!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()