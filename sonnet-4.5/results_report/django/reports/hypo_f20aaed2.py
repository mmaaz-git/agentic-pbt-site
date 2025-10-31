from hypothesis import given, strategies as st
from django.core.cache.utils import make_template_fragment_key

@given(
    st.lists(st.text(min_size=1), min_size=2, max_size=5)
)
def test_no_separator_collision(vary_on_list):
    joined_with_separator = ':'.join(vary_on_list)

    key1 = make_template_fragment_key('test', vary_on_list)
    key2 = make_template_fragment_key('test', [joined_with_separator])

    if vary_on_list != [joined_with_separator]:
        assert key1 != key2, f"Collision: {vary_on_list} produces same key as [{joined_with_separator}]"

# Run the test
if __name__ == "__main__":
    test_no_separator_collision()