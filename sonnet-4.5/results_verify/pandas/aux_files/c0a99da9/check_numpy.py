import numpy as np
import inspect

# Check actual numpy.argsort signature
sig = inspect.signature(np.argsort)
print("numpy.argsort signature:")
print(sig)
print("\nDefault values:")
for param_name, param in sig.parameters.items():
    if param.default != inspect.Parameter.empty:
        print(f"  {param_name}: {param.default!r}")