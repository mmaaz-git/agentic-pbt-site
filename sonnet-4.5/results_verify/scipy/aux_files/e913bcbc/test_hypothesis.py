from hypothesis import given, strategies as st
import scipy.constants as sc


def test_find_none_returns_all_keys():
    result = sc.find(None)
    expected = sorted(sc.physical_constants.keys())
    if result == expected:
        print("✓ test_find_none_returns_all_keys PASSED")
    else:
        print("✗ test_find_none_returns_all_keys FAILED")
        print(f"  Result has {len(result)} keys")
        print(f"  Expected has {len(expected)} keys")
        print(f"  Missing {len(expected) - len(result)} keys")
        # Find some missing keys
        missing = set(expected) - set(result)
        print(f"  Example missing keys: {list(missing)[:5]}")


@given(st.sampled_from(list(sc.physical_constants.keys())))
def test_find_can_locate_all_constants(key):
    results = sc.find(key)
    assert key in results, f"find() could not locate key '{key}' that exists in physical_constants"


# Run the first test
test_find_none_returns_all_keys()

# Run the property test with a few examples
print("\nTesting individual constants:")
failed = []
for key in ['Planck constant over 2 pi', 'reduced Planck constant', 'atomic unit of velocity']:
    try:
        results = sc.find(key)
        if key in results:
            print(f"✓ Found '{key}'")
        else:
            print(f"✗ Cannot find '{key}'")
            failed.append(key)
    except Exception as e:
        print(f"✗ Error with '{key}': {e}")
        failed.append(key)

if failed:
    print(f"\nFailed to find {len(failed)} constants that exist in physical_constants")