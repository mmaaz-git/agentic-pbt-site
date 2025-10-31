from django.dispatch import Signal


def simple_receiver(**kwargs):
    return "received"


# Test case 1: sender=None with use_caching=True
print("Test 1: sender=None with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=None)
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 2: sender="test_string" with use_caching=True
print("\nTest 2: sender='test_string' with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender="test_string")
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 3: sender=123 with use_caching=True
print("\nTest 3: sender=123 with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=123)
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 4: sender=object() with use_caching=True
print("\nTest 4: sender=object() with use_caching=True")
try:
    signal = Signal(use_caching=True)
    signal.connect(simple_receiver)
    signal.send(sender=object())
    print("SUCCESS: No error occurred")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 5: All cases work with use_caching=False
print("\nTest 5: All sender types with use_caching=False")
try:
    signal = Signal(use_caching=False)
    signal.connect(simple_receiver)
    signal.send(sender=None)
    signal.send(sender="test_string")
    signal.send(sender=123)
    signal.send(sender=object())
    print("SUCCESS: All sender types work without caching")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")