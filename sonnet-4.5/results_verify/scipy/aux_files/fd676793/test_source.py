import scipy.optimize
import inspect

# Get the source code of newton
print("Newton function source (first 100 lines):")
source = inspect.getsource(scipy.optimize.newton)
lines = source.split('\n')
for i, line in enumerate(lines[:100], 1):
    if 'rtol' in line or 'tol' in line and 'ValueError' in line:
        print(f"{i:3}: {line}")
    elif 'def newton' in line:
        print(f"{i:3}: {line}")
    elif 'ValueError' in line:
        print(f"{i:3}: {line}")

print("\n\nBisect function source (checking validation):")
source = inspect.getsource(scipy.optimize.bisect)
lines = source.split('\n')
for i, line in enumerate(lines[:50], 1):
    if 'rtol' in line:
        print(f"{i:3}: {line}")