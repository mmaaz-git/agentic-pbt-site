import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.utils.datastructures import CaseInsensitiveMapping

cim = CaseInsensitiveMapping({'ß': 'value'})

print("Testing German eszett (ß):")
print(f"cim.get('ß') = {cim.get('ß')}")
print(f"cim.get('SS') = {cim.get('SS')}")
print(f"cim.get('ss') = {cim.get('ss')}")

print("\nTrying cim['SS']:")
try:
    result = cim['SS']
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")

print("\nTrying cim['ss']:")
try:
    result = cim['ss']
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")

print("\n--- Checking case transformations ---")
print(f"'ß'.upper() = '{('ß').upper()}'")
print(f"'ß'.lower() = '{('ß').lower()}'")
print(f"'SS'.lower() = '{('SS').lower()}'")
print(f"'ss'.upper() = '{('ss').upper()}'")

print("\n--- Checking casefold transformations ---")
print(f"'ß'.casefold() = '{('ß').casefold()}'")
print(f"'SS'.casefold() = '{('SS').casefold()}'")
print(f"'ss'.casefold() = '{('ss').casefold()}'")