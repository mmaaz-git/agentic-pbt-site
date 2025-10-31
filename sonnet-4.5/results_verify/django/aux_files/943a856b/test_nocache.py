from django.dispatch import Signal

# Test without caching
signal = Signal(use_caching=False)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

print(f"Without caching - has_listeners(): {signal.has_listeners()}")
print(f"Without caching - send(sender=None): {signal.send(sender=None)}")
print(f"Without caching - send(sender=object()): {signal.send(sender=object())}")

# Test with caching
signal2 = Signal(use_caching=True)
signal2.connect(receiver, weak=False)

try:
    signal2.has_listeners()
    print("With caching - has_listeners() succeeded")
except TypeError as e:
    print(f"With caching - has_listeners() failed: {e}")

try:
    signal2.send(sender=None)
    print("With caching - send(sender=None) succeeded")
except TypeError as e:
    print(f"With caching - send(sender=None) failed: {e}")

try:
    signal2.send(sender=object())
    print("With caching - send(sender=object()) succeeded")
except TypeError as e:
    print(f"With caching - send(sender=object()) failed: {e}")