import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st
from django.core.signing import b62_encode, b62_decode

@given(st.text())
def test_b62_decode_does_not_crash(s):
    try:
        result = b62_decode(s)
    except (IndexError, ValueError):
        assert False, f"b62_decode should not crash on input: {s!r}"

if __name__ == "__main__":
    test_b62_decode_does_not_crash()