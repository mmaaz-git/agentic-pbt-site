#!/usr/bin/env python
import sys
import traceback
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.10/site-packages')

from django.dispatch import Signal

print("Testing to get exact traceback...")
print("=" * 70)

try:
    signal = Signal(use_caching=True)

    def my_receiver(signal, sender, **kwargs):
        return "received"

    signal.connect(my_receiver, sender=None, weak=False)
    result = signal.send(sender=None)
    print("Success! Result:", result)
except Exception as e:
    print(f"Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()