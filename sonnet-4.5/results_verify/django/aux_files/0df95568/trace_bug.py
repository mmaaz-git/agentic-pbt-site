import traceback
from django.dispatch import Signal

signal = Signal(use_caching=True)
sender = object()

def receiver(**kwargs):
    return 42

signal.connect(receiver, sender=sender, weak=False)

try:
    result = signal.has_listeners(sender=sender)
except TypeError as e:
    print("Full traceback:")
    traceback.print_exc()
    print("\nError message:", e)
