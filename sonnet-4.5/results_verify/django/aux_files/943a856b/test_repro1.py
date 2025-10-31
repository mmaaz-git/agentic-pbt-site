from django.dispatch import Signal

signal = Signal(use_caching=True)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

try:
    result = signal.has_listeners()
    print(f"has_listeners() returned: {result}")
except TypeError as e:
    print(f"TypeError occurred: {e}")