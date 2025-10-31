#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.10/site-packages')

from hypothesis import given, strategies as st, settings
from django.dispatch import Signal

@settings(max_examples=500)
@given(st.booleans())
def test_cache_clearing_on_connect(use_caching):
    signal = Signal(use_caching=use_caching)
    sender = object()

    def receiver1(**kwargs):
        return 1

    signal.connect(receiver1, sender=sender, weak=False)
    signal.send(sender)

    def receiver2(**kwargs):
        return 2

    signal.connect(receiver2, sender=sender, weak=False)
    responses = signal.send(sender)

    assert len(responses) == 2

print("Running hypothesis test...")
print("=" * 70)

try:
    test_cache_clearing_on_connect()
    print("✓ Test passed for all examples!")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
except TypeError as e:
    print(f"✗ TypeError occurred: {e}")
    print("\nThis confirms the bug - test fails with use_caching=True")
except Exception as e:
    print(f"✗ Unexpected error: {type(e).__name__}: {e}")