import pandas as pd


class TestClass:
    pass


# Test 1: Using instance with methodtype="classmethod" (as in bug report)
print("Test 1: Using instance with methodtype='classmethod'")
try:
    err = pd.errors.AbstractMethodError(TestClass(), methodtype="classmethod")
    print(f"Success: {str(err)}")
except AttributeError as e:
    print(f"Error: {e}")

# Test 2: Using class with methodtype="classmethod" (as in pandas docs example)
print("\nTest 2: Using class with methodtype='classmethod'")
try:
    err = pd.errors.AbstractMethodError(TestClass, methodtype="classmethod")
    print(f"Success: {str(err)}")
except AttributeError as e:
    print(f"Error: {e}")

# Test 3: Using instance with methodtype="method" (default)
print("\nTest 3: Using instance with methodtype='method'")
try:
    err = pd.errors.AbstractMethodError(TestClass())
    print(f"Success: {str(err)}")
except AttributeError as e:
    print(f"Error: {e}")

# Test 4: All other methodtype values with instance
for methodtype in ["staticmethod", "property"]:
    print(f"\nTest with methodtype='{methodtype}' and instance:")
    try:
        err = pd.errors.AbstractMethodError(TestClass(), methodtype=methodtype)
        print(f"Success: {str(err)}")
    except AttributeError as e:
        print(f"Error: {e}")