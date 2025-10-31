from hypothesis import given, strategies as st, assume
import pandas as pd

# Property-based test from bug report
@given(st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=5), min_size=2, max_size=20),
       st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=5), min_size=1, max_size=10))
def test_categorical_add_remove_roundtrip(initial_categories, new_categories):
    assume(len(set(initial_categories)) == len(initial_categories))
    assume(len(set(new_categories)) == len(new_categories))
    assume(not any(cat in initial_categories for cat in new_categories))

    cat = pd.Categorical(['a'], categories=initial_categories)
    original_categories = list(cat.categories)

    cat_with_added = cat.add_categories(new_categories)
    cat_removed = cat_with_added.remove_categories(new_categories)

    assert list(cat_removed.categories) == original_categories

# Test with the failing input
print("Testing with the specific failing input:")
initial_categories = ['z', 'y', 'x']
new_categories = ['a']

cat = pd.Categorical(['a'], categories=initial_categories)
original_categories = list(cat.categories)
print(f"Original categories: {original_categories}")

cat_with_added = cat.add_categories(new_categories)
print(f"After adding {new_categories}: {list(cat_with_added.categories)}")

cat_removed = cat_with_added.remove_categories(new_categories)
print(f"After removing {new_categories}: {list(cat_removed.categories)}")

if list(cat_removed.categories) == original_categories:
    print("✓ Test passed: Categories preserved after round-trip")
else:
    print(f"✗ Test failed: Expected {original_categories}, got {list(cat_removed.categories)}")

# Run the hypothesis test
print("\nRunning hypothesis test...")
try:
    test_categorical_add_remove_roundtrip()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")