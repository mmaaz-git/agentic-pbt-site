from pandas.util.version import InfinityType, NegativeInfinityType

# Test InfinityType instances
inf1 = InfinityType()
inf2 = InfinityType()

print("Testing InfinityType:")
print(f"inf1 == inf2: {inf1 == inf2}")
print(f"inf1 > inf2: {inf1 > inf2}")
print(f"inf1 < inf2: {inf1 < inf2}")
print(f"inf1 <= inf2: {inf1 <= inf2}")
print(f"inf1 >= inf2: {inf1 >= inf2}")

# Mathematical consistency check for InfinityType
print("\nVerifying mathematical consistency for InfinityType:")
if inf1 == inf2:
    print("inf1 equals inf2")
    if inf1 > inf2:
        print("ERROR: inf1 > inf2 is True when they are equal!")
    if inf1 < inf2:
        print("ERROR: inf1 < inf2 is True when they are equal!")
    if not (inf1 <= inf2):
        print("ERROR: inf1 <= inf2 is False when they are equal!")
    if not (inf1 >= inf2):
        print("ERROR: inf1 >= inf2 is False when they are equal!")

# Test NegativeInfinityType instances
ninf1 = NegativeInfinityType()
ninf2 = NegativeInfinityType()

print("\nTesting NegativeInfinityType:")
print(f"ninf1 == ninf2: {ninf1 == ninf2}")
print(f"ninf1 > ninf2: {ninf1 > ninf2}")
print(f"ninf1 < ninf2: {ninf1 < ninf2}")
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")
print(f"ninf1 >= ninf2: {ninf1 >= ninf2}")

# Mathematical consistency check for NegativeInfinityType
print("\nVerifying mathematical consistency for NegativeInfinityType:")
if ninf1 == ninf2:
    print("ninf1 equals ninf2")
    if ninf1 > ninf2:
        print("ERROR: ninf1 > ninf2 is True when they are equal!")
    if ninf1 < ninf2:
        print("ERROR: ninf1 < ninf2 is True when they are equal!")
    if not (ninf1 <= ninf2):
        print("ERROR: ninf1 <= ninf2 is False when they are equal!")
    if not (ninf1 >= ninf2):
        print("ERROR: ninf1 >= ninf2 is False when they are equal!")