import pandas as pd
import pandas.api.interchange as interchange
from hypothesis import given, strategies as st, settings, assume


@given(st.lists(st.sampled_from(['a', 'b', 'c']), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=200)
def test_from_dataframe_categorical_with_nulls(cat_values, null_mask):
    assume(len(cat_values) == len(null_mask))

    values_with_nulls = [None if null else val for val, null in zip(cat_values, null_mask)]
    df = pd.DataFrame({'cat': pd.Categorical(values_with_nulls)})

    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False, check_categorical=False)

# Test with the specific failing input mentioned
print("Testing with specific failing input: cat_values=['a', 'a'], null_mask=[False, True]")
try:
    test_from_dataframe_categorical_with_nulls(['a', 'a'], [False, True])
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed with assertion error: {e}")
except Exception as e:
    print(f"Test failed with error: {e}")

# Run the hypothesis test
print("\nRunning hypothesis test...")
try:
    test_from_dataframe_categorical_with_nulls()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")