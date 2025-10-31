#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
import struct

# Test the reported bug
values = [4.484782386619779e-144]
encoded = llm.encode(values)
decoded = llm.decode(encoded)

print(f"Original: {values[0]}")
print(f"Decoded:  {decoded[0]}")
print(f"Match: {values[0] == decoded[0]}")

# Additional tests to understand the behavior
print("\n--- Additional Analysis ---")
print(f"Encoded bytes: {encoded.hex()}")
print(f"Encoded length: {len(encoded)} bytes")

# Check what happens with float32 min values
import sys
print(f"\nFloat32 min positive normal: {sys.float_info.min}")
print(f"Float64 min positive: {sys.float_info.min}")

# Test with various small values
test_values = [
    1.0e-38,  # Near float32 normal range
    1.0e-45,  # Near float32 denormal min
    1.0e-100, # Below float32 range
    1.0e-200, # Way below float32 range
    4.484782386619779e-144  # Original failing value
]

print("\nTesting various small values:")
for val in test_values:
    enc = llm.encode([val])
    dec = llm.decode(enc)
    print(f"  {val:20e} -> {dec[0]:20e} (match: {val == dec[0]})")