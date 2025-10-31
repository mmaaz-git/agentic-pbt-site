import math
from anyio import CapacityLimiter

# Show that float('inf') and math.inf are equal but not identical
print(f"float('inf') == math.inf: {float('inf') == math.inf}")
print(f"float('inf') is math.inf: {float('inf') is math.inf}")
print()

# Try to create CapacityLimiter with float('inf')
print("Creating CapacityLimiter with float('inf'):")
try:
    limiter = CapacityLimiter(float('inf'))
    print("Success - limiter created")
except TypeError as e:
    print(f"Error: {e}")
print()

# Try to create CapacityLimiter with math.inf
print("Creating CapacityLimiter with math.inf:")
try:
    limiter = CapacityLimiter(math.inf)
    print("Success - limiter created")
except TypeError as e:
    print(f"Error: {e}")