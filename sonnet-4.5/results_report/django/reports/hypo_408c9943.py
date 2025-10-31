#!/usr/bin/env python3
"""
Property-based test using Hypothesis that reveals the Django Azerbaijani bidi bug.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import django.conf.locale as locale_module
from hypothesis import given, strategies as st


def is_rtl_script(text):
    """Check if text contains RTL script characters."""
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0780, 0x07BF),  # Thaana
        (0x07C0, 0x07FF),  # N'Ko
        (0x0800, 0x083F),  # Samaritan
        (0x0840, 0x085F),  # Mandaic
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
        (0xFB50, 0xFDFF),  # Arabic presentation forms A
        (0xFE70, 0xFEFF),  # Arabic presentation forms B
    ]

    return any(
        any(start <= ord(c) <= end for start, end in rtl_ranges)
        for c in text
    )


@given(st.sampled_from(list(locale_module.LANG_INFO.keys())))
def test_bidi_languages_use_rtl_script(lang_code):
    """Test that languages marked as bidi use RTL scripts."""
    info = locale_module.LANG_INFO[lang_code]

    if info.get('bidi', False):
        name_local = info.get('name_local', '')

        assert is_rtl_script(name_local), \
            f"Language {lang_code} marked as bidi but name_local '{name_local}' " \
            f"doesn't use RTL script"


if __name__ == '__main__':
    # Run the test
    test_bidi_languages_use_rtl_script()