from django.dispatch import Signal

# Test the bug report
signal = Signal(use_caching=True)
sender = object()

def receiver(**kwargs):
    return 42

signal.connect(receiver, sender=sender, weak=False)

try:
    result = signal.has_listeners(sender=sender)
    print(f"has_listeners succeeded: {result}")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    
# Also test with use_caching=False
signal2 = Signal(use_caching=False)
sender2 = object()
signal2.connect(receiver, sender=sender2, weak=False)
result2 = signal2.has_listeners(sender=sender2)
print(f"has_listeners with use_caching=False: {result2}")
