from pandas.util.version import InfinityType, NegativeInfinityType

inf1 = InfinityType()
inf2 = InfinityType()

print(f"inf1 == inf2: {inf1 == inf2}")
print(f"inf1 > inf2: {inf1 > inf2}")
print(f"inf1 < inf2: {inf1 < inf2}")
print(f"inf1 <= inf2: {inf1 <= inf2}")
print(f"inf1 >= inf2: {inf1 >= inf2}")

assert inf1 == inf2
assert inf1 > inf2

ninf1 = NegativeInfinityType()
ninf2 = NegativeInfinityType()

print(f"\nninf1 == ninf2: {ninf1 == ninf2}")
print(f"ninf1 < ninf2: {ninf1 < ninf2}")
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")

assert ninf1 == ninf2
assert ninf1 < ninf2