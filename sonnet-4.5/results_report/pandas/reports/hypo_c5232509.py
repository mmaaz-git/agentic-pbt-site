from hypothesis import given, strategies as st, example
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["cat1", "cat2", "cat3", None]), min_size=0, max_size=100),
    st.booleans()
)
@example(['cat1', None], False)  # Add the specific failing case
def test_round_trip_categorical(cat_list, ordered):
    df = pd.DataFrame({"col": pd.Categorical(cat_list, ordered=ordered)})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)


# Run the test
if __name__ == "__main__":
    print("Running property-based tests...")
    # Run the property-based test
    test_round_trip_categorical()