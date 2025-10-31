import numpy.char as char
import inspect

# Get documentation for each function
functions = [char.find, char.rfind, char.startswith, char.endswith]

for func in functions:
    print(f"\n{'='*60}")
    print(f"Function: numpy.char.{func.__name__}")
    print('='*60)
    print(func.__doc__)
    print("\nSignature:")
    print(inspect.signature(func))