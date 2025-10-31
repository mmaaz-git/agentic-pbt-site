import pandas as pd
from hypothesis import given, settings, strategies as st

# First, let's run the simple reproduction case
print("=" * 50)
print("Simple reproduction case:")
print("=" * 50)

df = pd.DataFrame({
    'key': pd.Categorical([2, 0], categories=[5, 4, 3, 2, 1, 0], ordered=True),
    'value': [10, 20]
})

result_sorted = df.groupby('key', observed=False, sort=True).sum()
print("With sort=True:")
print(f"Index: {list(result_sorted.index)}")
print(f"Values:\n{result_sorted}")

result_unsorted = df.groupby('key', observed=False, sort=False).sum()
print("\nWith sort=False:")
print(f"Index: {list(result_unsorted.index)}")
print(f"Values:\n{result_unsorted}")

# Now let's run the specific failing case from hypothesis
print("\n" + "=" * 50)
print("Hypothesis failing case: keys=[0], values=[0.0]")
print("=" * 50)

keys = [0]
values = [0.0]
categories_ordered = [5, 4, 3, 2, 1, 0]

df2 = pd.DataFrame({
    'key': pd.Categorical(keys, categories=categories_ordered, ordered=True),
    'value': values
})

result_sort_true = df2.groupby('key', observed=False, sort=True).sum()
result_sort_false = df2.groupby('key', observed=False, sort=False).sum()

print(f"sort=True index: {list(result_sort_true.index)}")
print(f"sort=False index: {list(result_sort_false.index)}")

print(f"\nExpected order: {categories_ordered}")
print(f"sort=True matches expected: {list(result_sort_true.index) == categories_ordered}")
print(f"sort=False matches expected: {list(result_sort_false.index) == categories_ordered}")

# Let's also test with unordered categorical
print("\n" + "=" * 50)
print("Testing with unordered categorical:")
print("=" * 50)

df3 = pd.DataFrame({
    'key': pd.Categorical([2, 0], categories=[5, 4, 3, 2, 1, 0], ordered=False),
    'value': [10, 20]
})

result_unordered_sorted = df3.groupby('key', observed=False, sort=True).sum()
print("Unordered categorical with sort=True:")
print(f"Index: {list(result_unordered_sorted.index)}")

result_unordered_unsorted = df3.groupby('key', observed=False, sort=False).sum()
print("\nUnordered categorical with sort=False:")
print(f"Index: {list(result_unordered_unsorted.index)}")

# Full hypothesis test
print("\n" + "=" * 50)
print("Running full hypothesis test:")
print("=" * 50)

@given(
    st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=20),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=20),
)
@settings(max_examples=10)
def test_categorical_groupby_sort_false_with_observed_false(keys, values):
    if len(keys) != len(values):
        return

    categories_ordered = [5, 4, 3, 2, 1, 0]

    df = pd.DataFrame({
        'key': pd.Categorical(keys, categories=categories_ordered, ordered=True),
        'value': values
    })

    result_sort_true = df.groupby('key', observed=False, sort=True).sum()
    result_sort_false = df.groupby('key', observed=False, sort=False).sum()

    expected_order = categories_ordered

    assert list(result_sort_true.index) == expected_order, \
        f"sort=True failed: {list(result_sort_true.index)} != {expected_order}"

    assert list(result_sort_false.index) == expected_order, \
        f"sort=False failed: {list(result_sort_false.index)} != {expected_order}, keys={keys}"

try:
    test_categorical_groupby_sort_false_with_observed_false()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")