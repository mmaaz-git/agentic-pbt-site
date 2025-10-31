from hypothesis import given, strategies as st, settings as hyp_settings
from django.conf import settings
from django.conf.urls.static import static

settings.configure(DEBUG=True)

@hyp_settings(max_examples=200)
@given(st.integers(min_value=1, max_value=100))
def test_static_slash_only_prefix_bug(num_slashes):
    prefix = '/' * num_slashes
    result = static(prefix)

    if result:
        pattern_obj = result[0].pattern
        regex_pattern = pattern_obj.regex.pattern

        stripped = prefix.lstrip('/')

        if not stripped:
            assert regex_pattern != r'^(?P<path>.*)$', \
                f"BUG: Slash-only prefix '{prefix}' creates catch-all regex: {regex_pattern}"

# Run the test
test_static_slash_only_prefix_bug()