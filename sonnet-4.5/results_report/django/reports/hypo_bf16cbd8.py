#!/usr/bin/env python3
"""
Property-based test using Hypothesis to find bugs in Django's ConnectionHandler.configure_settings.
This test checks for idempotence - running configure_settings twice should produce the same result.
"""

from hypothesis import given, strategies as st, settings
from django.db.utils import ConnectionHandler
import copy

@given(st.one_of(
    st.just({}),
    st.dictionaries(
        st.just('default'),
        st.dictionaries(
            st.sampled_from(['ENGINE', 'NAME', 'USER', 'PASSWORD', 'HOST', 'PORT', 'OPTIONS', 'TEST']),
            st.one_of(st.text(), st.dictionaries(st.text(), st.text()), st.booleans(), st.integers()),
            max_size=5
        ),
        min_size=1,
        max_size=1
    ).flatmap(lambda default_db: st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(lambda x: x != 'default'),
        st.dictionaries(
            st.sampled_from(['ENGINE', 'NAME', 'USER']),
            st.text(max_size=50),
            max_size=3
        ),
        max_size=5
    ).map(lambda other_dbs: {**default_db, **other_dbs}))
))
@settings(max_examples=500)
def test_configure_settings_idempotence(databases):
    handler = ConnectionHandler()
    configured_once = handler.configure_settings(copy.deepcopy(databases))
    configured_twice = handler.configure_settings(copy.deepcopy(configured_once))
    assert configured_once == configured_twice

if __name__ == "__main__":
    # Run the test
    test_configure_settings_idempotence()