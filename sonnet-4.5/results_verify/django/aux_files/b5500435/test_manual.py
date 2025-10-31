from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


print("Testing Signal with use_caching=True and sender=None")
signal = Signal(use_caching=True)
signal.connect(simple_receiver)
try:
    signal.send(sender=None)
    print("SUCCESS: No error with sender=None")
except TypeError as e:
    print(f"ERROR with sender=None: {e}")

print("\nTesting with sender='test_string'")
try:
    signal.send(sender="test_string")
    print("SUCCESS: No error with sender='test_string'")
except TypeError as e:
    print(f"ERROR with sender='test_string': {e}")

print("\nTesting with sender=123")
try:
    signal.send(sender=123)
    print("SUCCESS: No error with sender=123")
except TypeError as e:
    print(f"ERROR with sender=123: {e}")

print("\nTesting with sender=object()")
try:
    signal.send(sender=object())
    print("SUCCESS: No error with sender=object()")
except TypeError as e:
    print(f"ERROR with sender=object(): {e}")

print("\n\nTesting Signal with use_caching=False")
signal2 = Signal(use_caching=False)
signal2.connect(simple_receiver)
try:
    signal2.send(sender=None)
    print("SUCCESS: No error with use_caching=False and sender=None")
except TypeError as e:
    print(f"ERROR with use_caching=False and sender=None: {e}")