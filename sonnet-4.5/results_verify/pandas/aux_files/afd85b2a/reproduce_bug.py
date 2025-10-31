import pandas.util.version as pv

neg_inf = pv.NegativeInfinity

print(f"neg_inf == neg_inf: {neg_inf == neg_inf}")
print(f"neg_inf < neg_inf: {neg_inf < neg_inf}")
print(f"neg_inf > neg_inf: {neg_inf > neg_inf}")
print(f"neg_inf <= neg_inf: {neg_inf <= neg_inf}")
print(f"neg_inf >= neg_inf: {neg_inf >= neg_inf}")
print(f"neg_inf != neg_inf: {neg_inf != neg_inf}")

print("\nChecking the problem:")
print(f"NegativeInfinity.__eq__ returns: {neg_inf.__eq__(neg_inf)}")
print(f"NegativeInfinity.__lt__ returns: {neg_inf.__lt__(neg_inf)}")

print("\nThis violates the trichotomy law!")
print("When neg_inf == neg_inf is True, neg_inf < neg_inf should be False")

assert neg_inf == neg_inf
assert not (neg_inf < neg_inf)