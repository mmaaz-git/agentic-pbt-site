import pandas.compat
import inspect

print("Documentation says:")
print(pandas.compat.is_platform_power.__doc__)

print("\nImplementation:")
print(inspect.getsource(pandas.compat.is_platform_power))

print("\nActual function name: is_platform_power")
print("Checking for: Power architecture (ppc64, ppc64le)")