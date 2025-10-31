from hypothesis import given, assume, strategies as st, example
import dask.dataframe.io.parquet.core as parquet_core

@given(
    st.lists(st.text(min_size=1), min_size=1, max_size=10),
    st.lists(st.dictionaries(
        st.text(min_size=1),
        st.one_of(st.none(), st.integers(), st.floats(allow_nan=False)),
        min_size=0,
        max_size=10
    ), min_size=1, max_size=10)
)
@example(parts=['part1'], statistics=[{}])  # Force the specific failing input
def test_apply_filters_returns_subset(parts, statistics):
    assume(len(parts) == len(statistics))
    for stats in statistics:
        stats['columns'] = []
    filters = []
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    assert len(out_parts) <= len(parts)

if __name__ == "__main__":
    # Run the test to find failures
    import traceback
    try:
        test_apply_filters_returns_subset()
        print("Test passed (no failures found in default runs)")
    except Exception as e:
        print(f"Test failed!")
        print(f"Exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nFailing input: parts=['part1'], statistics=[{}], filters=[]")