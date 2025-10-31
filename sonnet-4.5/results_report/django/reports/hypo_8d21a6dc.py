#!/usr/bin/env python3
"""Hypothesis property-based test for Django i18n GetLanguageInfoListNode bug"""

import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    LANGUAGE_CODE='en-us',
    USE_I18N=True,
    USE_L10N=True,
)
django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=2))
def test_get_language_info_handles_sequences(language):
    node = GetLanguageInfoListNode(None, 'result')

    try:
        result = node.get_language_info(language)
    except TypeError as e:
        raise AssertionError(
            f"Should handle sequence {language} but got TypeError: {e}"
        )

if __name__ == "__main__":
    test_get_language_info_handles_sequences()