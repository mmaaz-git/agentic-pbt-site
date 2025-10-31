import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=True)

from hypothesis import given, strategies as st
from django.conf.locale import LANG_INFO

@given(st.sampled_from(list(LANG_INFO.keys())))
def test_language_variants_have_base_language(lang_code):
    if '-' in lang_code:
        base_lang = lang_code.split('-')[0]
        info = LANG_INFO[lang_code]

        if 'fallback' in info and not ('name' in info):
            return

        assert base_lang in LANG_INFO, \
            f"Language variant {lang_code} exists but base language {base_lang} is not in LANG_INFO"

if __name__ == "__main__":
    test_language_variants_have_base_language()