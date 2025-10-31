#!/usr/bin/env python3
"""Test with skip_repeats=True to see if it prevents the issue"""

from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant

# Test 1: Same lambda functions (different cache keys)
print("Test 1: Different lambda functions (different cache keys)")
dep1 = Dependant(call=lambda: "dep1", name="dep1")
dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])
dep1.dependencies.append(dep2)

print(f"dep1.cache_key: {dep1.cache_key}")
print(f"dep2.cache_key: {dep2.cache_key}")

try:
    flat = get_flat_dependant(dep1, skip_repeats=True)
    print("SUCCESS with skip_repeats=True!")
    print(f"Result type: {type(flat)}")
except RecursionError:
    print("FAILED with RecursionError even with skip_repeats=True")

print("\n" + "="*60 + "\n")

# Test 2: Same function (same cache key)
print("Test 2: Same function (same cache key)")
def shared_func():
    return "shared"

dep3 = Dependant(call=shared_func, name="dep3")
dep4 = Dependant(call=shared_func, name="dep4", dependencies=[dep3])
dep3.dependencies.append(dep4)

print(f"dep3.cache_key: {dep3.cache_key}")
print(f"dep4.cache_key: {dep4.cache_key}")
print(f"Are cache keys equal? {dep3.cache_key == dep4.cache_key}")

try:
    flat = get_flat_dependant(dep3, skip_repeats=True)
    print("SUCCESS with skip_repeats=True and same cache key!")
    print(f"Result type: {type(flat)}")
except RecursionError:
    print("FAILED with RecursionError even with skip_repeats=True and same cache key")