from hypothesis import given, strategies as st, settings
import dask.bag as db


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=20),
       st.integers(min_value=1, max_value=10))
def test_from_sequence_respects_npartitions(seq, npartitions):
    bag = db.from_sequence(seq, npartitions=npartitions)

    expected = npartitions if len(seq) >= npartitions else len(seq)

    assert bag.npartitions == expected, \
        f"from_sequence(seq of len {len(seq)}, npartitions={npartitions}) " \
        f"produced {bag.npartitions} partitions, expected {expected}"


if __name__ == "__main__":
    test_from_sequence_respects_npartitions()