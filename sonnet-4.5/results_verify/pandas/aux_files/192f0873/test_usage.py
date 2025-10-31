import pandas.util.version as version_module

# Test how these sentinel values are used in sorting/comparison contexts
print("=== Testing usage in sorting context ===")

# Create a list with regular values and infinity values
values = [10, version_module.Infinity, 5, version_module.NegativeInfinity, 20, version_module.Infinity]

print("Original list:", values)
try:
    sorted_values = sorted(values)
    print("Sorted list:", sorted_values)
except Exception as e:
    print(f"Error sorting: {e}")

# Test with version comparison keys
print("\n=== Testing in version comparison context ===")

# Look at how these are used as sentinel values in _cmpkey
def test_version_comparison():
    v1 = version_module.Version("1.0.0")
    v2 = version_module.Version("1.0.0a1")  # Pre-release
    v3 = version_module.Version("1.0.0.post1")  # Post-release
    v4 = version_module.Version("1.0.0.dev1")  # Dev release

    print(f"v1 key: {v1._key}")
    print(f"v2 key (pre-release): {v2._key}")
    print(f"v3 key (post-release): {v3._key}")
    print(f"v4 key (dev-release): {v4._key}")

    # Test sorting
    versions = [v1, v2, v3, v4]
    sorted_versions = sorted(versions)
    print("\nSorted versions:")
    for v in sorted_versions:
        print(f"  {v}")

test_version_comparison()

# Test the specific case where Infinity is compared with itself
print("\n=== Testing tuple comparison with Infinity ===")
inf = version_module.Infinity
neg_inf = version_module.NegativeInfinity

# Simulating what happens in version key comparison
tuple1 = (1, (1, 0, 0), inf, neg_inf, inf, neg_inf)  # Regular release
tuple2 = (1, (1, 0, 0), inf, neg_inf, inf, neg_inf)  # Same release

print(f"tuple1: {tuple1}")
print(f"tuple2: {tuple2}")
print(f"tuple1 == tuple2: {tuple1 == tuple2}")
print(f"tuple1 < tuple2: {tuple1 < tuple2}")
print(f"tuple1 > tuple2: {tuple1 > tuple2}")