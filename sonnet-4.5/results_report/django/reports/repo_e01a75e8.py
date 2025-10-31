from django.dispatch import Signal

# Create a signal with caching enabled
signal = Signal(use_caching=True)

# Define a simple receiver function
def receiver(sender, **kwargs):
    return "received"

# Connect the receiver to the signal
signal.connect(receiver)

# Try to send a signal with sender=None (documented as valid)
# This should work but will crash due to the WeakKeyDictionary
print("Attempting to send signal with sender=None...")
try:
    result = signal.send(sender=None)
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")

# Also try with a plain object instance
print("\nAttempting to send signal with sender=object()...")
try:
    result = signal.send(sender=object())
    print(f"Success: {result}")
except TypeError as e:
    print(f"Error: {e}")

# Show that has_listeners() also crashes
print("\nAttempting to check has_listeners()...")
try:
    has_listeners = signal.has_listeners()
    print(f"Has listeners: {has_listeners}")
except TypeError as e:
    print(f"Error: {e}")

# Demonstrate that it works fine without caching
print("\n--- Testing without caching ---")
signal_no_cache = Signal(use_caching=False)
signal_no_cache.connect(receiver)

print("Sending with sender=None (no caching)...")
result = signal_no_cache.send(sender=None)
print(f"Success: {result}")

print("Sending with sender=object() (no caching)...")
result = signal_no_cache.send(sender=object())
print(f"Success: {result}")