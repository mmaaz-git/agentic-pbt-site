from anyio import CapacityLimiter
import math

print("Test 1: integer value")
try:
    limiter = CapacityLimiter(5)
    print(f"✓ Accepted 5 (int): total_tokens={limiter.total_tokens}")
except Exception as e:
    print(f"✗ Rejected 5: {e}")

print("\nTest 2: math.inf")
try:
    limiter = CapacityLimiter(math.inf)
    print(f"✓ Accepted math.inf: total_tokens={limiter.total_tokens}")
except Exception as e:
    print(f"✗ Rejected math.inf: {e}")

print("\nTest 3: float 5.5")
try:
    limiter = CapacityLimiter(5.5)
    print(f"✓ Accepted 5.5 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 5.5: {e}")

print("\nTest 4: float 1.5")
try:
    limiter = CapacityLimiter(1.5)
    print(f"✓ Accepted 1.5 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 1.5: {e}")

print("\nTest 5: float 10.25")
try:
    limiter = CapacityLimiter(10.25)
    print(f"✓ Accepted 10.25 (float): total_tokens={limiter.total_tokens}")
except TypeError as e:
    print(f"✗ Rejected 10.25: {e}")