from django.dispatch import Signal

# Case 1: has_listeners() with default sender=None
print("Test Case 1: has_listeners() with default sender=None")
print("-" * 50)

signal = Signal(use_caching=True)

def receiver(**kwargs):
    return "response"

signal.connect(receiver, weak=False)

try:
    result = signal.has_listeners()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 2: send() with sender=None
print("Test Case 2: send() with sender=None")
print("-" * 50)

signal2 = Signal(use_caching=True)

def receiver2(**kwargs):
    return "response"

signal2.connect(receiver2, weak=False)

try:
    result = signal2.send(sender=None)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 3: send() with sender=object()
print("Test Case 3: send() with sender=object()")
print("-" * 50)

signal3 = Signal(use_caching=True)

def receiver3(**kwargs):
    return "response"

signal3.connect(receiver3, weak=False)

try:
    result = signal3.send(sender=object())
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 50 + "\n")

# Case 4: Control - same operations with use_caching=False work fine
print("Test Case 4: Control - same operations with use_caching=False")
print("-" * 50)

signal4 = Signal(use_caching=False)

def receiver4(**kwargs):
    return "response"

signal4.connect(receiver4, weak=False)

try:
    result1 = signal4.has_listeners()
    print(f"has_listeners(): {result1}")

    result2 = signal4.send(sender=None)
    print(f"send(sender=None): {result2}")

    result3 = signal4.send(sender=object())
    print(f"send(sender=object()): {result3}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")