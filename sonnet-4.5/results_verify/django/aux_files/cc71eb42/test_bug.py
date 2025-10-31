import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django.conf.locale as locale_module
from hypothesis import given, strategies as st


def is_rtl_script(text):
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0780, 0x07BF),  # Thaana
        (0x07C0, 0x07FF),  # NKo
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
    info = locale_module.LANG_INFO[lang_code]

    if info.get('bidi', False):
        name_local = info.get('name_local', '')

        assert is_rtl_script(name_local), \
            f"Language {lang_code} marked as bidi but name_local '{name_local}' " \
            f"doesn't use RTL script"

# Run the test
if __name__ == "__main__":
    # Test specifically for 'az'
    print("Testing Azerbaijani ('az'):")
    try:
        test_bidi_languages_use_rtl_script('az')
        print("Test passed for 'az'")
    except AssertionError as e:
        print(f"Test failed for 'az': {e}")

    print("\nTesting all bidi languages:")
    for lang_code, info in locale_module.LANG_INFO.items():
        if info.get('bidi', False):
            try:
                test_bidi_languages_use_rtl_script(lang_code)
                print(f"✓ {lang_code}: {info['name_local']} - Test passed")
            except AssertionError as e:
                print(f"✗ {lang_code}: {info['name_local']} - Test failed")