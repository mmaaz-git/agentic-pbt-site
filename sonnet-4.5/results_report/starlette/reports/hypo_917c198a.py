from hypothesis import given, strategies as st, assume, settings
from starlette.responses import FileResponse, MalformedRangeHeader, RangeNotSatisfiable


def generate_valid_range_str(file_size):
    ranges_strategy = st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=file_size-1),
            st.integers(min_value=0, max_value=file_size-1)
        ).map(lambda r: (min(r), max(r)) if r[0] != r[1] else (r[0], r[0]+1 if r[0] < file_size - 1 else r[0])),
        min_size=1,
        max_size=10
    )
    return ranges_strategy.map(
        lambda ranges: "bytes=" + ",".join(f"{s}-{e-1 if e > s else e}" for s, e in ranges)
    )


@given(st.integers(min_value=100, max_value=10000).flatmap(
    lambda fs: st.tuples(st.just(fs), generate_valid_range_str(fs))
))
@settings(max_examples=500)
def test_parsed_ranges_are_non_overlapping(args):
    file_size, http_range = args

    try:
        result = FileResponse._parse_range_header(http_range, file_size)
    except (MalformedRangeHeader, RangeNotSatisfiable):
        assume(False)
        return

    for i in range(len(result) - 1):
        start1, end1 = result[i]
        start2, end2 = result[i+1]
        assert end1 <= start2, (
            f"Ranges overlap or are not sorted: "
            f"range {i} = ({start1}, {end1}), "
            f"range {i+1} = ({start2}, {end2}), "
            f"input = {http_range}"
        )


if __name__ == "__main__":
    test_parsed_ranges_are_non_overlapping()