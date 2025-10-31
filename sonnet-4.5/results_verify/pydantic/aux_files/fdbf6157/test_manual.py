import math
from anyio.abc import CapacityLimiter

limiter = CapacityLimiter(10)

limiter.total_tokens = 5
print(f"Integer works: {limiter.total_tokens}")

limiter.total_tokens = math.inf
print(f"math.inf works: {limiter.total_tokens}")

try:
    limiter.total_tokens = 2.5
    print(f"Float 2.5 works: {limiter.total_tokens}")
except TypeError as e:
    print(f"Error with float 2.5: {e}")