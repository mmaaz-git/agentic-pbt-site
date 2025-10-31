import math
from anyio.abc import CapacityLimiter

limiter = CapacityLimiter(10)  # Use integer for initialization

limiter.total_tokens = 5
print(f"Integer works: {limiter.total_tokens}")

limiter.total_tokens = math.inf
print(f"math.inf works: {limiter.total_tokens}")

limiter.total_tokens = 2.5
print(f"Float should work but fails: {limiter.total_tokens}")