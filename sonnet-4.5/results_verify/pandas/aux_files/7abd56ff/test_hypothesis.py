from hypothesis import given, strategies as st, example
from pandas.io.formats.printing import _justify


@given(
    st.lists(st.lists(st.text(max_size=10), min_size=1, max_size=5), min_size=1, max_size=5),
    st.lists(st.lists(st.text(max_size=10), min_size=1, max_size=5), min_size=1, max_size=5)
)
@example([['', '']], [['']])  # First failing example from report
@example([['']], [['', '']])  # Second failing example from report
def test_justify_preserves_content(head, tail):
    """
    Property: _justify should preserve all content from head and tail.
    """
    result_head, result_tail = _justify(head, tail)

    for i, (orig, justified) in enumerate(zip(head, result_head)):
        assert len(justified) == len(orig), \
            f"head[{i}] length changed: {len(orig)} -> {len(justified)}"

    for i, (orig, justified) in enumerate(zip(tail, result_tail)):
        assert len(justified) == len(orig), \
            f"tail[{i}] length changed: {len(orig)} -> {len(justified)}"

if __name__ == "__main__":
    # Run the test
    test_justify_preserves_content()