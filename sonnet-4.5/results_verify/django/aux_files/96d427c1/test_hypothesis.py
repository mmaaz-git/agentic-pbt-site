import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, example
from django.utils.datastructures import CaseInsensitiveMapping


@given(st.dictionaries(st.text(), st.text()))
@settings(max_examples=500)
@example({'ß': ''})  # Add the specific failing example
def test_case_insensitive_mapping_access(d):
    cim = CaseInsensitiveMapping(d)
    for key, value in d.items():
        try:
            assert cim.get(key) == value
            assert cim.get(key.upper()) == value
            assert cim.get(key.lower()) == value
        except AssertionError as e:
            print(f"Failed on key='{key}', value='{value}'")
            print(f"  cim.get('{key}') = {cim.get(key)}")
            print(f"  cim.get('{key.upper()}') = {cim.get(key.upper())}")
            print(f"  cim.get('{key.lower()}') = {cim.get(key.lower())}")
            print(f"  Expected: {value}")
            raise

if __name__ == "__main__":
    test_case_insensitive_mapping_access()
    print("Test completed")