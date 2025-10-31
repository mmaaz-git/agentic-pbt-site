from hypothesis import given, strategies as st, settings, Verbosity
import pandas as pd

@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10))
@settings(verbosity=Verbosity.verbose, max_examples=10)
def test_categorical_preserves_missing_values(categories):
    """Categorical -1 codes (missing values) should round-trip as NaN."""
    cat_data = pd.Categorical.from_codes([-1], categories=list(set(categories)), ordered=False)
    series = pd.Series(cat_data)

    from pandas.core.interchange.from_dataframe import from_dataframe
    df = pd.DataFrame({"cat": series})
    result = from_dataframe(df.__dataframe__(allow_copy=True))

    assert pd.isna(result["cat"].iloc[0]), f"Missing categorical value not preserved for categories={list(set(categories))}"

if __name__ == "__main__":
    test_categorical_preserves_missing_values()