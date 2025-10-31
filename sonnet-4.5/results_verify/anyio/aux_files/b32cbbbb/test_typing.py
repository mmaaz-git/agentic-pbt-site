#!/usr/bin/env python3
"""Test to understand Python typing behavior with float annotation"""

from typing import get_type_hints
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env')

from anyio import create_memory_object_stream

# Get type hints for the function
hints = get_type_hints(create_memory_object_stream.__new__)
print("Type hints for create_memory_object_stream.__new__:")
for key, value in hints.items():
    print(f"  {key}: {value}")

# Test what Python considers a float
test_values = [1.0, 1, 5.5, 0.0, 0, float('inf')]
for val in test_values:
    print(f"\nTesting {val} (type: {type(val).__name__}):")
    print(f"  isinstance({val}, float): {isinstance(val, float)}")
    print(f"  isinstance({val}, int): {isinstance(val, int)}")