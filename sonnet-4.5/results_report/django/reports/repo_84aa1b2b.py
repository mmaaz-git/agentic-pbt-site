import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=True)

from django.conf.locale import LANG_INFO
from django.utils.translation import get_language_info

print("1. zh-hans exists:", 'zh-hans' in LANG_INFO)
print("2. zh-hant exists:", 'zh-hant' in LANG_INFO)
print("3. zh exists:", 'zh' in LANG_INFO)

try:
    info = get_language_info('zh')
    print(f"4. get_language_info('zh') returned: {info}")
except KeyError as e:
    print(f"4. get_language_info('zh') fails: {e}")

try:
    info = get_language_info('zh-unknown')
    print(f"5. get_language_info('zh-unknown') returned: {info}")
except KeyError as e:
    print(f"5. get_language_info('zh-unknown') fails: {e}")

print("\nComparing with other language variants that work correctly:")
print("6. en exists:", 'en' in LANG_INFO)
print("7. en-gb exists:", 'en-gb' in LANG_INFO)

try:
    info = get_language_info('en-unknown')
    print(f"8. get_language_info('en-unknown') returned: code={info.get('code')}, name={info.get('name')}")
except KeyError as e:
    print(f"8. get_language_info('en-unknown') fails: {e}")