import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO


def test_azerbaijani_bidi_bug():
    az_info = LANG_INFO['az']

    assert az_info['bidi'] == False, (
        f"Azerbaijani (az) is incorrectly marked as bidi=True. "
        f"Azerbaijani has used Latin script since 1991 and is a left-to-right language."
    )

if __name__ == "__main__":
    test_azerbaijani_bidi_bug()
    print("Test passed!")