import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

# Get Azerbaijani locale info
az_info = LANG_INFO['az']

# Display the current configuration
print("Azerbaijani locale configuration:")
print(f"  Code: {az_info['code']}")
print(f"  Name: {az_info['name']}")
print(f"  Name Local: {az_info['name_local']}")
print(f"  Bidi (Right-to-Left): {az_info['bidi']}")

print("\nComparing with other RTL languages:")
for lang_code in ['ar', 'fa', 'he', 'ur']:
    lang_info = LANG_INFO[lang_code]
    print(f"  {lang_code}: name_local='{lang_info['name_local']}' (bidi={lang_info['bidi']})")

print("\nThe Issue:")
print("Azerbaijani is marked as bidi=True (right-to-left) but uses Latin script.")
print("The name_local 'Azərbaycanca' is written in Latin characters.")
print("All other RTL languages use actual RTL scripts (Arabic, Hebrew, etc.).")

# Demonstrate the error
assert az_info['bidi'] == True, "Expected bidi=True (current state)"
print("\nCurrent assertion passes: az_info['bidi'] == True")

# What it should be
print("\nWhat it should be: az_info['bidi'] == False")
print("Because Azerbaijani uses Latin script which is left-to-right.")