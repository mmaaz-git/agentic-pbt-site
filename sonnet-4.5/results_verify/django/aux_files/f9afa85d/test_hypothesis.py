from hypothesis import given, strategies as st, settings
from django.core.cache.utils import make_template_fragment_key

@given(
    st.lists(st.text(min_size=1), min_size=2, max_size=5)
)
@settings(max_examples=100)
def test_no_separator_collision(vary_on_list):
    joined_with_separator = ':'.join(vary_on_list)

    key1 = make_template_fragment_key('test', vary_on_list)
    key2 = make_template_fragment_key('test', [joined_with_separator])

    if vary_on_list != [joined_with_separator]:
        assert key1 != key2, f"Collision found: {vary_on_list} produces same key as [{joined_with_separator}]"

# Run the test
try:
    test_no_separator_collision()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")