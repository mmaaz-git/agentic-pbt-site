#!/usr/bin/env python3
"""Test to reproduce the bug report for anyio.create_memory_object_stream"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env')

from anyio import create_memory_object_stream

# Test 1: Try with float value 1.0
print("Test 1: Calling create_memory_object_stream(max_buffer_size=1.0)")
try:
    send, receive = create_memory_object_stream(max_buffer_size=1.0)
    print("SUCCESS: Created stream with max_buffer_size=1.0")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

# Test 2: Try with float value 5.5
print("\nTest 2: Calling create_memory_object_stream(max_buffer_size=5.5)")
try:
    send, receive = create_memory_object_stream(max_buffer_size=5.5)
    print("SUCCESS: Created stream with max_buffer_size=5.5")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

# Test 3: Try with integer value 1
print("\nTest 3: Calling create_memory_object_stream(max_buffer_size=1)")
try:
    send, receive = create_memory_object_stream(max_buffer_size=1)
    print("SUCCESS: Created stream with max_buffer_size=1")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

# Test 4: Try with math.inf
import math
print("\nTest 4: Calling create_memory_object_stream(max_buffer_size=math.inf)")
try:
    send, receive = create_memory_object_stream(max_buffer_size=math.inf)
    print("SUCCESS: Created stream with max_buffer_size=math.inf")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

# Test 5: Try with float value 0.0
print("\nTest 5: Calling create_memory_object_stream(max_buffer_size=0.0)")
try:
    send, receive = create_memory_object_stream(max_buffer_size=0.0)
    print("SUCCESS: Created stream with max_buffer_size=0.0")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")