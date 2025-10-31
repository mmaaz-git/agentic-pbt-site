from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

print("Testing AlwaysGreaterThan:")
print("=" * 40)
inf1 = AlwaysGreaterThan()
inf2 = AlwaysGreaterThan()

print(f"inf1 == inf2: {inf1 == inf2}")  # Should be True
print(f"inf1 != inf2: {inf1 != inf2}")  # Should be False
print(f"inf1 > inf2: {inf1 > inf2}")    # Should be False when equal (BUG: returns True)
print(f"inf1 < inf2: {inf1 < inf2}")    # Should be False when equal
print(f"inf1 >= inf2: {inf1 >= inf2}")  # Should be True when equal
print(f"inf1 <= inf2: {inf1 <= inf2}")  # Should be True when equal (BUG: returns False)

print("\nTesting AlwaysLessThan:")
print("=" * 40)
ninf1 = AlwaysLessThan()
ninf2 = AlwaysLessThan()

print(f"ninf1 == ninf2: {ninf1 == ninf2}")  # Should be True
print(f"ninf1 != ninf2: {ninf1 != ninf2}")  # Should be False
print(f"ninf1 < ninf2: {ninf1 < ninf2}")    # Should be False when equal (BUG: returns True)
print(f"ninf1 > ninf2: {ninf1 > ninf2}")    # Should be False when equal
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")  # Should be True when equal
print(f"ninf1 >= ninf2: {ninf1 >= ninf2}")  # Should be True when equal (BUG: returns False)

print("\nTesting comparisons with other values:")
print("=" * 40)
print(f"inf1 > 100: {inf1 > 100}")         # Should be True
print(f"inf1 > 'string': {inf1 > 'string'}")  # Should be True
print(f"ninf1 < 100: {ninf1 < 100}")       # Should be True
print(f"ninf1 < 'string': {ninf1 < 'string'}")  # Should be True