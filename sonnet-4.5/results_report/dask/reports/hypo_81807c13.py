from hypothesis import given, strategies as st, settings


@st.composite
def task_dict_strategy(draw):
    num_pairs = draw(st.integers(min_value=0, max_value=10))
    pairs = []
    for _ in range(num_pairs):
        key = draw(st.text(min_size=1, max_size=10))
        value = draw(st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ))
        pairs.append([key, value])
    return (dict, pairs)


@given(task_dict_strategy())
@settings(max_examples=200)
def test_unquote_dict_no_crash(task):
    from dask.diagnostics.profile_visualize import unquote
    try:
        result = unquote(task)
        assert isinstance(result, dict)
    except IndexError:
        raise AssertionError(f"unquote crashed with IndexError on input: {task}")


if __name__ == "__main__":
    test_unquote_dict_no_crash()