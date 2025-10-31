from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

print("Testing dask tokenization with different OPTIONS settings:")
print("=" * 60)

# Get token with use_new_combine_kwarg_defaults=False
with set_options(use_new_combine_kwarg_defaults=False):
    token1 = obj.__dask_tokenize__()
    hash1 = hash(obj)
    print(f"With use_new=False:")
    print(f"  Token: {token1}")
    print(f"  Hash: {hash1}")

# Get token with use_new_combine_kwarg_defaults=True
with set_options(use_new_combine_kwarg_defaults=True):
    token2 = obj.__dask_tokenize__()
    hash2 = hash(obj)
    print(f"\nWith use_new=True:")
    print(f"  Token: {token2}")
    print(f"  Hash: {hash2}")

print("\n" + "=" * 60)
print(f"Tokens are equal: {token1 == token2}")
print(f"Hashes are equal: {hash1 == hash2}")

if token1 != token2:
    print("\nERROR: Dask tokens changed when OPTIONS changed!")
    print("This violates dask's requirement that tokens must be deterministic.")

if hash1 != hash2:
    print("\nERROR: Hash values changed when OPTIONS changed!")
    print("This violates Python's requirement that hash must be constant during an object's lifetime.")