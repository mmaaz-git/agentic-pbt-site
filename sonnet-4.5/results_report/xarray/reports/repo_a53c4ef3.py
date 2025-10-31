from xarray.util.deprecation_helpers import deprecate_dims


@deprecate_dims
def example_func(*, dim=None):
    return dim


# Test case: both dims and dim provided
# According to the bug report, the deprecated parameter (dims) overwrites the new one (dim)
result = example_func(dims="x", dim="y")

print(f"Expected: 'y' (new parameter should take precedence)")
print(f"Got: {result!r}")

# Verify the issue
if result == "x":
    print("\n⚠️ BUG CONFIRMED: deprecated parameter 'dims' is overwriting new parameter 'dim'")
else:
    print("\n✓ OK: new parameter 'dim' takes precedence")
