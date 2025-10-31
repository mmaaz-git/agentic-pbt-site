import pandas.util.version as version_module

# Test 1: Reproducing the bug as shown in the report
print("=== Reproducing the bug ===")
inf = version_module.Infinity
neg_inf = version_module.NegativeInfinity

print(f"Infinity > Infinity: {inf > inf}")
print(f"Infinity < Infinity: {inf < inf}")
print(f"Infinity == Infinity: {inf == inf}")
print(f"Infinity >= Infinity: {inf >= inf}")
print(f"Infinity <= Infinity: {inf <= inf}")
print()
print(f"NegativeInfinity < NegativeInfinity: {neg_inf < neg_inf}")
print(f"NegativeInfinity > NegativeInfinity: {neg_inf > neg_inf}")
print(f"NegativeInfinity == NegativeInfinity: {neg_inf == neg_inf}")
print(f"NegativeInfinity >= NegativeInfinity: {neg_inf >= neg_inf}")
print(f"NegativeInfinity <= NegativeInfinity: {neg_inf <= neg_inf}")

print("\n=== Testing comparisons with other values ===")
print(f"Infinity > 10: {inf > 10}")
print(f"Infinity > 'string': {inf > 'string'}")
print(f"10 < Infinity: {10 < inf}")
print(f"NegativeInfinity < 10: {neg_inf < 10}")
print(f"NegativeInfinity < 'string': {neg_inf < 'string'}")
print(f"10 > NegativeInfinity: {10 > neg_inf}")

# Test 2: Property-based test from the report
print("\n=== Running property-based tests ===")

def test_infinity_self_comparison():
    inf = version_module.Infinity

    try:
        assert not (inf < inf), "FAIL: Infinity < Infinity should be False"
        print("PASS: Infinity < Infinity is False")
    except AssertionError as e:
        print(e)

    try:
        assert not (inf > inf), "FAIL: Infinity > Infinity should be False"
        print("PASS: Infinity > Infinity is False")
    except AssertionError as e:
        print(e)

    try:
        assert inf == inf, "FAIL: Infinity == Infinity should be True"
        print("PASS: Infinity == Infinity is True")
    except AssertionError as e:
        print(e)

    try:
        assert inf <= inf, "FAIL: Infinity <= Infinity should be True"
        print("PASS: Infinity <= Infinity is True")
    except AssertionError as e:
        print(e)

    try:
        assert inf >= inf, "FAIL: Infinity >= Infinity should be True"
        print("PASS: Infinity >= Infinity is True")
    except AssertionError as e:
        print(e)

def test_negative_infinity_self_comparison():
    neg_inf = version_module.NegativeInfinity

    try:
        assert not (neg_inf < neg_inf), "FAIL: NegativeInfinity < NegativeInfinity should be False"
        print("PASS: NegativeInfinity < NegativeInfinity is False")
    except AssertionError as e:
        print(e)

    try:
        assert not (neg_inf > neg_inf), "FAIL: NegativeInfinity > NegativeInfinity should be False"
        print("PASS: NegativeInfinity > NegativeInfinity is False")
    except AssertionError as e:
        print(e)

    try:
        assert neg_inf == neg_inf, "FAIL: NegativeInfinity == NegativeInfinity should be True"
        print("PASS: NegativeInfinity == NegativeInfinity is True")
    except AssertionError as e:
        print(e)

    try:
        assert neg_inf <= neg_inf, "FAIL: NegativeInfinity <= NegativeInfinity should be True"
        print("PASS: NegativeInfinity <= NegativeInfinity is True")
    except AssertionError as e:
        print(e)

    try:
        assert neg_inf >= neg_inf, "FAIL: NegativeInfinity >= NegativeInfinity should be True"
        print("PASS: NegativeInfinity >= NegativeInfinity is True")
    except AssertionError as e:
        print(e)

test_infinity_self_comparison()
print()
test_negative_infinity_self_comparison()