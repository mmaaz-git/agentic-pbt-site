#!/usr/bin/env python3
"""Property-based test for ModelFormMixin.get_success_url() bug"""

from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin


# Strategy for generating URL templates with placeholders
url_template_with_placeholders = st.builds(
    lambda prefix, placeholder, suffix: f"{prefix}{{{placeholder}}}{suffix}",
    prefix=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='/-')),
    placeholder=st.text(min_size=1, max_size=15, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='_')),
    suffix=st.text(min_size=0, max_size=20, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='/-'))
)

@given(success_url_template=url_template_with_placeholders)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_modelformmixin_should_not_raise_confusing_keyerror(success_url_template):
    mixin = ModelFormMixin()
    mixin.success_url = success_url_template
    mock_obj = Mock()
    mock_obj.__dict__ = {}  # Empty dict means placeholder won't be found
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
    except KeyError as e:
        raise AssertionError(
            f"get_success_url() should not raise KeyError for URL {success_url_template!r}. "
            "It should either validate the template or provide a helpful error message."
        )

if __name__ == "__main__":
    # Run the test
    test_modelformmixin_should_not_raise_confusing_keyerror()