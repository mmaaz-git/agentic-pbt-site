#!/usr/bin/env python3
"""Reproduce the reported bug in anyio.create_memory_object_stream"""

import math
from anyio import create_memory_object_stream

print("Testing create_memory_object_stream with different input types:")
print("=" * 60)

# Test 1: Float value (5.5)
print("\n1. Testing with float 5.5:")
try:
    send, recv = create_memory_object_stream(5.5)
    print("   ✓ Accepted 5.5")
    send.close()
    recv.close()
except ValueError as e:
    print(f"   ✗ Rejected 5.5: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

# Test 2: Integer value (5)
print("\n2. Testing with integer 5:")
try:
    send, recv = create_memory_object_stream(5)
    print("   ✓ Accepted 5 (int)")
    send.close()
    recv.close()
except ValueError as e:
    print(f"   ✗ Rejected 5: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

# Test 3: math.inf
print("\n3. Testing with math.inf:")
try:
    send, recv = create_memory_object_stream(math.inf)
    print("   ✓ Accepted math.inf")
    send.close()
    recv.close()
except ValueError as e:
    print(f"   ✗ Rejected math.inf: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

# Test 4: Other float values
test_values = [0.5, 3.14, 100.1, 0.0, 1.0]
print("\n4. Testing various float values:")
for val in test_values:
    try:
        send, recv = create_memory_object_stream(val)
        print(f"   ✓ Accepted {val}")
        send.close()
        recv.close()
    except ValueError as e:
        print(f"   ✗ Rejected {val}: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error for {val}: {e}")

# Test 5: Run the hypothesis test
print("\n5. Running property-based test with hypothesis:")
print("-" * 40)

from hypothesis import given, strategies as st

@given(st.floats(min_value=0, max_value=1000).filter(
    lambda x: x != math.inf and not (isinstance(x, float) and x == int(x)) and not math.isnan(x)
))
def test_max_buffer_size_type_contract(value):
    """
    Test that max_buffer_size accepts all non-negative floats as per type annotation.
    The type annotation says 'float', so all floats should be valid.
    """
    send_stream, receive_stream = create_memory_object_stream(value)
    send_stream.close()
    receive_stream.close()

try:
    test_max_buffer_size_type_contract()
    print("   ✓ Hypothesis test passed - all floats accepted")
except Exception as e:
    print(f"   ✗ Hypothesis test failed: {e}")
    # Show a specific failing example
    print("\n   Example failure case:")
    try:
        test_val = 5.5
        send, recv = create_memory_object_stream(test_val)
        print(f"   {test_val} was accepted (unexpected)")
    except ValueError as ve:
        print(f"   {test_val} was rejected: {ve}")