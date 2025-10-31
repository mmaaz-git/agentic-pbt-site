import numpy.char as char
import numpy as np

# Check the docstring of numpy.char.replace
print("=== numpy.char.replace docstring ===")
print(char.replace.__doc__)
print("\n" + "="*50 + "\n")

# Check if there's any documentation about dtype behavior
print("=== numpy.char.array docstring ===")
print(char.array.__doc__)
print("\n" + "="*50 + "\n")

# Check the module documentation
print("=== numpy.char module docstring (first 2000 chars) ===")
print(char.__doc__[:2000] if char.__doc__ else "No module docstring")