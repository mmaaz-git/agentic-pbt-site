import math
from anyio._core._synchronization import CapacityLimiter

print("Test 1: Creating CapacityLimiter with float 1.5")
try:
    limiter = CapacityLimiter(1.5)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 2: Creating CapacityLimiter with float 1.0")
try:
    limiter = CapacityLimiter(1.0)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 3: Creating CapacityLimiter with int 1")
try:
    limiter = CapacityLimiter(1)
    print(f"Success: Created limiter with total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 4: Setting total_tokens to float 2.5")
try:
    limiter = CapacityLimiter(1)
    limiter.total_tokens = 2.5
    print(f"Success: Set total_tokens to {limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")

print("\nTest 5: Setting total_tokens to math.inf")
try:
    limiter = CapacityLimiter(1)
    limiter.total_tokens = math.inf
    print(f"Success: Set total_tokens to {limiter.total_tokens}")
except TypeError as e:
    print(f"Failed with TypeError: {e}")