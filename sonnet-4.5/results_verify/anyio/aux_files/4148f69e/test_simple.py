import anyio
import math

print("Testing create_memory_object_stream with float values...")

# Test with 2.5
print("\nTest 1: max_buffer_size=2.5")
try:
    send, recv = anyio.create_memory_object_stream(2.5)
    print(f"Success: Created stream with max_buffer_size=2.5")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with 1.5
print("\nTest 2: max_buffer_size=1.5")
try:
    send, recv = anyio.create_memory_object_stream(1.5)
    print(f"Success: Created stream with max_buffer_size=1.5")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with integer (should work)
print("\nTest 3: max_buffer_size=2 (integer)")
try:
    send, recv = anyio.create_memory_object_stream(2)
    print(f"Success: Created stream with max_buffer_size=2")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with math.inf (should work)
print("\nTest 4: max_buffer_size=math.inf")
try:
    send, recv = anyio.create_memory_object_stream(math.inf)
    print(f"Success: Created stream with max_buffer_size=math.inf")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with 0.0 (float but integer value)
print("\nTest 5: max_buffer_size=0.0")
try:
    send, recv = anyio.create_memory_object_stream(0.0)
    print(f"Success: Created stream with max_buffer_size=0.0")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with 3.0 (float but integer value)
print("\nTest 6: max_buffer_size=3.0")
try:
    send, recv = anyio.create_memory_object_stream(3.0)
    print(f"Success: Created stream with max_buffer_size=3.0")
except ValueError as e:
    print(f"ValueError: {e}")