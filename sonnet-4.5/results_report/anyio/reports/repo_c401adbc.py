import anyio

# Test 1: Creating a CapacityLimiter with a non-integer float value
print("Test 1: Creating CapacityLimiter with float 2.5")
try:
    limiter = anyio.CapacityLimiter(2.5)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Setting total_tokens property to a non-integer float value
print("Test 2: Setting total_tokens property to float 3.7")
try:
    limiter = anyio.CapacityLimiter(1)
    print(f"Created limiter with {limiter.total_tokens} tokens")
    limiter.total_tokens = 3.7
    print(f"SUCCESS: Updated limiter to {limiter.total_tokens} tokens")
except TypeError as e:
    print(f"FAILED when setting property: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Creating with integer (should work)
print("Test 3: Creating CapacityLimiter with integer 5")
try:
    limiter = anyio.CapacityLimiter(5)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Creating with math.inf (should work)
print("Test 4: Creating CapacityLimiter with math.inf")
import math
try:
    limiter = anyio.CapacityLimiter(math.inf)
    print(f"SUCCESS: Created limiter with {limiter.total_tokens} tokens")
except Exception as e:
    print(f"FAILED: {e}")