import scipy.datasets

print("=== scipy.datasets.clear_cache documentation ===")
print(scipy.datasets.clear_cache.__doc__)
print("\n=== Function signature ===")
import inspect
sig = inspect.signature(scipy.datasets.clear_cache)
print(f"Signature: {sig}")
print(f"\nParameters:")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")