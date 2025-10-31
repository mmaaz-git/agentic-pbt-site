import pandas as pd
import pandas.api.typing
from hypothesis import given, strategies as st

# First, test the basic reproduction case
print("=== Basic Reproduction Test ===")
nat1 = pandas.api.typing.NaTType()
nat2 = pandas.api.typing.NaTType()

print(f"nat1: {nat1}")
print(f"nat2: {nat2}")
print(f"nat1 is nat2: {nat1 is nat2}")
print(f"nat1 is pd.NaT: {nat1 is pd.NaT}")
print(f"nat2 is pd.NaT: {nat2 is pd.NaT}")
print(f"nat1 == nat2: {nat1 == nat2}")
print(f"nat1 == pd.NaT: {nat1 == pd.NaT}")

# Test NAType for comparison
print("\n=== NAType Comparison ===")
na1 = pandas.api.typing.NAType()
na2 = pandas.api.typing.NAType()

print(f"na1: {na1}")
print(f"na2: {na2}")
print(f"na1 is na2: {na1 is na2}")
print(f"na1 is pd.NA: {na1 is pd.NA}")
print(f"na2 is pd.NA: {na2 is pd.NA}")

# Test the property-based test
print("\n=== Property-Based Test ===")
@given(st.integers(min_value=1, max_value=100))
def test_nattype_singleton_property(n):
    instances = [pandas.api.typing.NaTType() for _ in range(n)]

    for i in range(len(instances)):
        for j in range(len(instances)):
            if instances[i] is not instances[j]:
                print(f"FAILED: instance {i} is not instance {j} (n={n})")
                return False
    return True

# Run a simple test with n=2
print("Testing with n=2:")
instances = [pandas.api.typing.NaTType() for _ in range(2)]
if instances[0] is instances[1]:
    print("PASS: All instances are the same object")
else:
    print("FAIL: Instances are different objects")

# Check what pd.NaT actually is
print("\n=== pd.NaT Analysis ===")
print(f"pd.NaT: {pd.NaT}")
print(f"type(pd.NaT): {type(pd.NaT)}")
print(f"pd.NaT.__class__: {pd.NaT.__class__}")

# Check if pd.NaT itself is a singleton
print("\n=== pd.NaT Singleton Check ===")
import pandas._libs.tslibs.nattype as nattype_module
print(f"pd.NaT is pd.NaT: {pd.NaT is pd.NaT}")
print(f"Has c_NaT? {hasattr(nattype_module, 'c_NaT')}")
if hasattr(nattype_module, 'NaT'):
    print(f"nattype_module.NaT is pd.NaT: {nattype_module.NaT is pd.NaT}")