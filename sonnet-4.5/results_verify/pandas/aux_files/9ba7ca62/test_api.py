import pandas as pd

print("Checking if hash_tuples is exposed in the public API:")
print("=" * 50)

# Check if it's in pandas.util
if hasattr(pd.util, 'hash_tuples'):
    print("✓ hash_tuples is available in pd.util")
else:
    print("✗ hash_tuples is NOT in pd.util")

# Check if it's in pandas.core.util.hashing (internal API)
try:
    from pandas.core.util.hashing import hash_tuples
    print("✓ hash_tuples is available in pandas.core.util.hashing (internal)")
except ImportError:
    print("✗ hash_tuples is NOT in pandas.core.util.hashing")

# List what's actually in pd.util
print("\nPublic API functions in pd.util:")
util_attrs = [attr for attr in dir(pd.util) if not attr.startswith('_')]
for attr in util_attrs:
    if 'hash' in attr.lower():
        print(f"  - {attr}")

# Check docstring if available
try:
    from pandas.core.util.hashing import hash_tuples
    print("\nDocstring for hash_tuples:")
    print("-" * 40)
    print(hash_tuples.__doc__)
except:
    pass