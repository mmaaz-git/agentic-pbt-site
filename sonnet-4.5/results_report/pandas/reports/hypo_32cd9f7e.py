from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(['a', 'b', 'c']), min_size=5, max_size=20),
    st.integers(min_value=0, max_value=4)
)
@settings(max_examples=100)
def test_categorical_preserves_missing(categories_list, null_idx):
    """Test that missing values in categorical columns are preserved during interchange conversion."""
    # Create codes array with a sentinel value (-1) for missing data
    codes = [0, 1, 2, -1, 0]
    cat = pd.Categorical.from_codes(codes, categories=['a', 'b', 'c'])
    df = pd.DataFrame({'cat': cat})

    # Convert through interchange protocol
    result = from_dataframe(df.__dataframe__())

    # Missing values should be preserved
    original_missing_count = df.isna().sum().sum()
    result_missing_count = result.isna().sum().sum()

    assert original_missing_count == result_missing_count, \
        f"Missing values not preserved: original had {original_missing_count}, result has {result_missing_count}"


if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis property-based test for categorical missing values...")
    print("=" * 70)

    try:
        test_categorical_preserves_missing()
        print("All tests passed!")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        print("\nFailing example details:")

        # Run a specific failing case to show details
        codes = [0, 1, 2, -1, 0]
        cat = pd.Categorical.from_codes(codes, categories=['a', 'b', 'c'])
        df = pd.DataFrame({'cat': cat})

        print(f"Input codes: {codes}")
        print(f"Categories: ['a', 'b', 'c']")
        print(f"\nOriginal DataFrame:")
        print(df)
        print(f"Missing values in original: {df.isna().sum().sum()}")

        result = from_dataframe(df.__dataframe__())
        print(f"\nAfter interchange conversion:")
        print(result)
        print(f"Missing values after conversion: {result.isna().sum().sum()}")

        print("\n" + "=" * 70)
        print("BUG CONFIRMED: Missing values are not preserved during interchange conversion!")
        print("The sentinel value -1 is incorrectly converted to a valid category value.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()