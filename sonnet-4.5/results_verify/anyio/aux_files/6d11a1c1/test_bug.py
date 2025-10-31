import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/anyio_env/lib/python3.13/site-packages')

# First test - the basic reproduction case
print("Test 1: Basic reproduction of the bug")
from anyio._core._synchronization import CapacityLimiterAdapter

try:
    limiter = CapacityLimiterAdapter(total_tokens=10)
    print(f"Created limiter with total_tokens=10")
    limiter.total_tokens = 5.5
    print(f"Successfully set total_tokens to 5.5")
except TypeError as e:
    print(f"TypeError occurred: {e}")

print()

# Test 2: Check if integers work
print("Test 2: Setting integer values")
try:
    limiter = CapacityLimiterAdapter(total_tokens=10)
    limiter.total_tokens = 5
    print(f"Successfully set total_tokens to 5 (int)")
    print(f"Current value: {limiter.total_tokens}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test 3: Check if math.inf works
print("Test 3: Setting math.inf")
import math
try:
    limiter = CapacityLimiterAdapter(total_tokens=10)
    limiter.total_tokens = math.inf
    print(f"Successfully set total_tokens to math.inf")
    print(f"Current value: {limiter.total_tokens}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test 4: Check asyncio backend
print("Test 4: Asyncio backend implementation")
from anyio._backends._asyncio import CapacityLimiter as AsyncioCapacityLimiter
try:
    limiter = AsyncioCapacityLimiter(total_tokens=10)
    limiter.total_tokens = 7.5
    print(f"Successfully set total_tokens to 7.5")
except TypeError as e:
    print(f"TypeError occurred: {e}")
