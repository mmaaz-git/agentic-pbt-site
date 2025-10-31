import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf.locale import LANG_INFO

# Reproduce the bug as shown in the report
print("=== Reproducing the Bug ===\n")

az_info = LANG_INFO['az']

print(f"Azerbaijani language info: {az_info}")
print(f"Marked as bidi (RTL): {az_info['bidi']}")
print(f"Name (local): {az_info['name_local']}")

print("\nOther bidi languages for comparison:")
for code in ['ar', 'fa', 'he', 'ur']:
    info = LANG_INFO[code]
    print(f"{code}: bidi={info['bidi']}, name_local='{info['name_local']}'")

print("\n=== Testing all bidi languages ===\n")

def is_rtl_script(text):
    """Check if text contains RTL script characters"""
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

# Test all languages marked as bidi
bidi_languages = [(code, info) for code, info in LANG_INFO.items() if info.get('bidi', False)]

print(f"Total languages marked as bidi: {len(bidi_languages)}\n")

for lang_code, info in bidi_languages:
    name_local = info.get('name_local', '')
    has_rtl = is_rtl_script(name_local)
    status = "✓" if has_rtl else "✗"
    print(f"{status} {lang_code:5} | {info['name']:30} | {name_local:20} | RTL script: {has_rtl}")

print("\n=== Character Analysis for Azerbaijani ===\n")
az_name = LANG_INFO['az']['name_local']
print(f"Azerbaijani name_local: '{az_name}'")
print("Character codes:")
for char in az_name:
    print(f"  '{char}' = U+{ord(char):04X} ({ord(char)})")

print("\n=== Historical Context ===")
print("Azerbaijani script history:")
print("- Until 1929: Arabic script (RTL)")
print("- 1939-1991: Cyrillic script (LTR)")
print("- 1991-present: Latin script (LTR)")
print("\nModern Azerbaijani uses Latin script with the following special characters:")
print("ə (schwa), ç, ğ, ı, ö, ş, ü")