import anyio

print("Testing with float 2.5:")
try:
    limiter = anyio.CapacityLimiter(2.5)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting property setter with 3.7:")
try:
    limiter = anyio.CapacityLimiter(1)
    print(f"Created limiter with {limiter.total_tokens} tokens")
    limiter.total_tokens = 3.7
    print(f"SUCCESS: Set total_tokens to {limiter.total_tokens}")
except TypeError as e:
    print(f"ERROR when setting to 3.7: {e}")

print("\nTesting with integer 5:")
try:
    limiter = anyio.CapacityLimiter(5)
    print(f"SUCCESS: Created limiter with integer {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting with math.inf:")
import math
try:
    limiter = anyio.CapacityLimiter(math.inf)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"ERROR: {e}")