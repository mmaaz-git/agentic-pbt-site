from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd


@given(
    df=data_frames(
        columns=[
            column("A", elements=st.sampled_from(["cat", "dog", "bird"])),
            column("B", elements=st.sampled_from(["x", "y", "z"])),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )
)
@settings(max_examples=200)
def test_get_dummies_from_dummies_with_drop_first(df):
    """
    Property: from_dummies should invert get_dummies(drop_first=True).
    Evidence: encoding.py line 376 states from_dummies "Inverts the operation
    performed by :func:`~pandas.get_dummies`."
    """
    dummies = pd.get_dummies(df, drop_first=True, dtype=int)

    default_cats = {}
    for col in df.columns:
        first_val = sorted(df[col].unique())[0]
        default_cats[col] = first_val

    recovered = pd.from_dummies(dummies, sep="_", default_category=default_cats)

    pd.testing.assert_frame_equal(recovered, df)

if __name__ == "__main__":
    test_get_dummies_from_dummies_with_drop_first()