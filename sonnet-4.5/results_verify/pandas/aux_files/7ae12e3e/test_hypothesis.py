from hypothesis import given, strategies as st
from pandas.io.formats.printing import adjoin


@given(st.lists(st.text(min_size=1), min_size=1), st.lists(st.text(min_size=1), min_size=1))
def test_adjoin_uses_strlen_consistently(list1, list2):
    call_count = {"count": 0}

    def counting_strlen(s):
        call_count["count"] += 1
        return len(s)

    adjoin(1, list1, list2, strlen=counting_strlen)

    total_strings = len(list1) + len(list2)
    assert call_count["count"] >= total_strings, f"Expected at least {total_strings} calls to strlen, but got {call_count['count']}"


if __name__ == "__main__":
    # Run the test
    test_adjoin_uses_strlen_consistently()
    print("Test completed")