#!/usr/bin/env python3

# First, test with the manual example
import pandas.io.json as json_module
import sys
import math

print("=== Manual Reproduction Test ===")
f = 1.7976931345e+308

print(f"sys.float_info.max: {sys.float_info.max}")
print(f"Original value: {f}")
print(f"Original is finite: {math.isfinite(f)}")
print(f"Original < sys.float_info.max: {f < sys.float_info.max}")

json_str = json_module.ujson_dumps(f)
recovered = json_module.ujson_loads(json_str)

print(f"JSON string: {json_str}")
print(f"Recovered: {recovered}")
print(f"Recovered is inf: {recovered == float('inf')}")
print(f"Recovered is finite: {math.isfinite(recovered)}")

# Now test with standard library for comparison
import json
print("\n=== Standard Library Comparison ===")
std_json_str = json.dumps(f)
std_recovered = json.loads(std_json_str)
print(f"Standard JSON string: {std_json_str}")
print(f"Standard recovered: {std_recovered}")
print(f"Standard recovered is finite: {math.isfinite(std_recovered)}")
print(f"Standard preserves value: {f == std_recovered}")

# Now run the Hypothesis test
print("\n=== Running Hypothesis Test ===")
from hypothesis import given, strategies as st, settings
import traceback

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_ujson_float_round_trip(f):
    """
    Property: Finite floats should round-trip through ujson
    """
    json_str = json_module.ujson_dumps(f)
    recovered = json_module.ujson_loads(json_str)
    assert math.isclose(f, recovered, rel_tol=1e-9, abs_tol=1e-15) or f == recovered, \
        f"Float round-trip failed: {f} -> {json_str} -> {recovered}"

try:
    test_ujson_float_round_trip()
    print("Hypothesis test passed all examples")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error in hypothesis test: {e}")
    traceback.print_exc()