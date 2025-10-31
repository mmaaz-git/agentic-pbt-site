import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

# Test the property-based test provided in the bug report
def test_azerbaijani_bidi_bug():
    az_info = LANG_INFO['az']

    print(f"Azerbaijani info: {az_info}")
    print(f"Azerbaijani bidi value: {az_info['bidi']}")

    assert az_info['bidi'] == False, (
        f"Azerbaijani (az) is incorrectly marked as bidi=True. "
        f"Azerbaijani has used Latin script since 1991 and is a left-to-right language."
    )

# Run the test
try:
    test_azerbaijani_bidi_bug()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed with: {e}")

# Check the reproduction code from the bug report
print("\n--- Reproduction from bug report ---")
az_info = LANG_INFO['az']
print(f"Azerbaijani bidi value: {az_info['bidi']}")

# This should pass according to the current implementation
assert az_info['bidi'] == True
print("Current assertion (bidi==True) passes")

# List all RTL languages for reference
print("\n--- All RTL (bidi=True) languages in Django ---")
for code, info in LANG_INFO.items():
    if isinstance(info, dict) and info.get('bidi') == True:
        print(f"  {code}: {info.get('name', 'N/A')} ({info.get('name_local', 'N/A')})")