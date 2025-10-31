from xarray.core.dtypes import AlwaysGreaterThan

# Create two instances of AlwaysGreaterThan
agt1 = AlwaysGreaterThan()
agt2 = AlwaysGreaterThan()

# Test equality - should be True
print(f"agt1 == agt2: {agt1 == agt2}")

# Test greater than - should be False since they're equal, but returns True (BUG!)
print(f"agt1 > agt2: {agt1 > agt2}")

# This violates the irreflexivity property
print(f"agt1 > agt1: {agt1 > agt1}")

# Test assertions
assert agt1 == agt2, "Expected two AlwaysGreaterThan instances to be equal"
assert agt1 > agt2, "This assertion passes but shouldn't - violates ordering properties!"