"""Test to reproduce the bug report for deep_dict_update"""
import copy
from hypothesis import given, strategies as st
from fastapi.openapi.utils import deep_dict_update

print("=" * 70)
print("REPRODUCING BUG REPORT")
print("=" * 70)

# First, test the hypothesis test case
def nested_dict_strategy(max_depth=3):
    if max_depth == 0:
        return st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text(), st.booleans(), st.none()),
            max_size=5
        )

    return st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.text(),
            st.booleans(),
            st.none(),
            st.lists(st.integers(), max_size=3),
            st.deferred(lambda: nested_dict_strategy(max_depth - 1))
        ),
        max_size=3
    )


@given(nested_dict_strategy(), nested_dict_strategy())
def test_deep_dict_update_idempotence(main, update):
    """Applying the same update twice should not change the result after first application"""
    main_copy1 = copy.deepcopy(main)
    main_copy2 = copy.deepcopy(main)

    deep_dict_update(main_copy1, update)
    result_after_first = copy.deepcopy(main_copy1)

    deep_dict_update(main_copy2, update)
    deep_dict_update(main_copy2, update)

    assert main_copy2 == result_after_first, f"Failed for main={main}, update={update}"


# Test the specific failing input mentioned
print("\n1. Testing specific failing input: main={}, update={'0': [0]}")
main = {}
update = {'0': [0]}
main_copy1 = copy.deepcopy(main)
main_copy2 = copy.deepcopy(main)

deep_dict_update(main_copy1, update)
result_after_first = copy.deepcopy(main_copy1)
print(f"   After 1st update: {main_copy1}")

deep_dict_update(main_copy2, update)
deep_dict_update(main_copy2, update)
print(f"   After 2nd update: {main_copy2}")
print(f"   Are they equal? {main_copy2 == result_after_first}")
print(f"   This {'PASSES' if main_copy2 == result_after_first else 'FAILS'} idempotence test")

# Test the manual reproduction case
print("\n2. Testing manual reproduction case with tags:")
main = {"tags": ["api"]}
update = {"tags": ["v1"]}

print(f"   Initial: {main}")
deep_dict_update(main, update)
print(f"   After 1st update: {main}")

deep_dict_update(main, update)
print(f"   After 2nd update: {main}")

deep_dict_update(main, update)
print(f"   After 3rd update: {main}")

# Test other data types for comparison
print("\n3. Testing idempotence with different data types:")

# Test with nested dicts (should be idempotent)
print("\n   a) Nested dicts:")
main = {"config": {"debug": True}}
update = {"config": {"verbose": False}}
print(f"      Initial: {main}")
deep_dict_update(main, update)
first = copy.deepcopy(main)
print(f"      After 1st: {main}")
deep_dict_update(main, update)
print(f"      After 2nd: {main}")
print(f"      Idempotent? {main == first}")

# Test with scalars (should be idempotent)
print("\n   b) Scalars:")
main = {"version": "1.0", "count": 5}
update = {"version": "2.0", "count": 10}
print(f"      Initial: {main}")
deep_dict_update(main, update)
first = copy.deepcopy(main)
print(f"      After 1st: {main}")
deep_dict_update(main, update)
print(f"      After 2nd: {main}")
print(f"      Idempotent? {main == first}")

# Test with lists (NOT idempotent with current implementation)
print("\n   c) Lists:")
main = {"items": [1, 2]}
update = {"items": [3, 4]}
print(f"      Initial: {main}")
deep_dict_update(main, update)
first = copy.deepcopy(main)
print(f"      After 1st: {main}")
deep_dict_update(main, update)
print(f"      After 2nd: {main}")
print(f"      Idempotent? {main == first}")

print("\n" + "=" * 70)
print("RUNNING HYPOTHESIS TEST")
print("=" * 70)

try:
    # Run a limited number of examples
    from hypothesis import settings
    test_deep_dict_update_idempotence_with_settings = settings(max_examples=50)(test_deep_dict_update_idempotence)
    test_deep_dict_update_idempotence_with_settings()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")