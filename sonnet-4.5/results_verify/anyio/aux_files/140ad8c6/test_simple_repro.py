from pandas.util.version import Infinity, NegativeInfinity

print("Testing Infinity comparisons:")
print(f"Infinity == Infinity: {Infinity == Infinity}")
print(f"Infinity <= Infinity: {Infinity <= Infinity}")
print(f"Infinity >= Infinity: {Infinity >= Infinity}")
print(f"Infinity < Infinity: {Infinity < Infinity}")
print(f"Infinity > Infinity: {Infinity > Infinity}")

print("\nTesting NegativeInfinity comparisons:")
print(f"NegativeInfinity == NegativeInfinity: {NegativeInfinity == NegativeInfinity}")
print(f"NegativeInfinity <= NegativeInfinity: {NegativeInfinity <= NegativeInfinity}")
print(f"NegativeInfinity >= NegativeInfinity: {NegativeInfinity >= NegativeInfinity}")
print(f"NegativeInfinity < NegativeInfinity: {NegativeInfinity < NegativeInfinity}")
print(f"NegativeInfinity > NegativeInfinity: {NegativeInfinity > NegativeInfinity}")

print("\nAssertion tests from the bug report:")
try:
    assert Infinity == Infinity
    print("✓ Infinity == Infinity passed")
except AssertionError:
    print("✗ Infinity == Infinity failed")

try:
    assert Infinity <= Infinity
    print("✓ Infinity <= Infinity passed")
except AssertionError:
    print("✗ Infinity <= Infinity failed")

try:
    assert NegativeInfinity == NegativeInfinity
    print("✓ NegativeInfinity == NegativeInfinity passed")
except AssertionError:
    print("✗ NegativeInfinity == NegativeInfinity failed")

try:
    assert NegativeInfinity >= NegativeInfinity
    print("✓ NegativeInfinity >= NegativeInfinity passed")
except AssertionError:
    print("✗ NegativeInfinity >= NegativeInfinity failed")

try:
    assert not (Infinity > Infinity)
    print("✓ not (Infinity > Infinity) passed")
except AssertionError:
    print("✗ not (Infinity > Infinity) failed")

try:
    assert not (NegativeInfinity < NegativeInfinity)
    print("✓ not (NegativeInfinity < NegativeInfinity) passed")
except AssertionError:
    print("✗ not (NegativeInfinity < NegativeInfinity) failed")