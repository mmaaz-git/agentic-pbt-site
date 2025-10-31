import pandas as pd

print("Testing pandas.eval() with empty and whitespace-only strings:\n")

# Test with empty string
print("pd.eval('') -> ", end="")
try:
    result = pd.eval('')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with single space
print("pd.eval(' ') -> ", end="")
try:
    result = pd.eval(' ')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with tab
print("pd.eval('\\t') -> ", end="")
try:
    result = pd.eval('\t')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with newline
print("pd.eval('\\n') -> ", end="")
try:
    result = pd.eval('\n')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with carriage return
print("pd.eval('\\r') -> ", end="")
try:
    result = pd.eval('\r')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with mixed whitespace
print("pd.eval('  \\t\\n  ') -> ", end="")
try:
    result = pd.eval('  \t\n  ')
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Also test the _check_expression function directly
print("\nTesting _check_expression directly:\n")
from pandas.core.computation.eval import _check_expression

# Test with empty string
print("_check_expression('') -> ", end="")
try:
    _check_expression('')
    print("No error raised")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with single space
print("_check_expression(' ') -> ", end="")
try:
    _check_expression(' ')
    print("No error raised")
except ValueError as e:
    print(f"ValueError: {e}")

# Test with tab
print("_check_expression('\\t') -> ", end="")
try:
    _check_expression('\t')
    print("No error raised")
except ValueError as e:
    print(f"ValueError: {e}")
