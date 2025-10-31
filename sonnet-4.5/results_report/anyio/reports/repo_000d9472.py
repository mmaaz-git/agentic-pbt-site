from anyio import create_memory_object_stream

# Test with float value 1.0
try:
    send, receive = create_memory_object_stream(max_buffer_size=1.0)
    print("Successfully created stream with max_buffer_size=1.0")
except ValueError as e:
    print(f"Error with max_buffer_size=1.0: {e}")

# Test with float value 5.5
try:
    send, receive = create_memory_object_stream(max_buffer_size=5.5)
    print("Successfully created stream with max_buffer_size=5.5")
except ValueError as e:
    print(f"Error with max_buffer_size=5.5: {e}")

# Test with float value 0.0
try:
    send, receive = create_memory_object_stream(max_buffer_size=0.0)
    print("Successfully created stream with max_buffer_size=0.0")
except ValueError as e:
    print(f"Error with max_buffer_size=0.0: {e}")

# Test with int value 1 (should work)
try:
    send, receive = create_memory_object_stream(max_buffer_size=1)
    print("Successfully created stream with max_buffer_size=1 (int)")
except ValueError as e:
    print(f"Error with max_buffer_size=1 (int): {e}")

# Test with math.inf (should work)
import math
try:
    send, receive = create_memory_object_stream(max_buffer_size=math.inf)
    print("Successfully created stream with max_buffer_size=math.inf")
except ValueError as e:
    print(f"Error with max_buffer_size=math.inf: {e}")