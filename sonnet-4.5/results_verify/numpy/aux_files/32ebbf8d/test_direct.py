#!/usr/bin/env python3
import numpy.f2py.symbolic as symbolic

print("Testing direct reproduction of bug...")
print("Input: \"'\" (single apostrophe)")

try:
    result = symbolic.eliminate_quotes("'")
    print(f"Unexpectedly succeeded: {result}")
except AssertionError as e:
    print(f"✓ Got expected AssertionError")
    print(f"  Error message: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting with other unmatched quotes...")
test_cases = ['"', "'test", 'test"', '"test\'', '\'test"']
for test in test_cases:
    print(f"\nTesting: {repr(test)}")
    try:
        result = symbolic.eliminate_quotes(test)
        print(f"  Result: {result}")
    except AssertionError as e:
        print(f"  AssertionError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")