from django.dispatch import Signal

# Create a signal with caching enabled
signal = Signal(use_caching=True)

# Define a receiver function
def receiver(sender, **kwargs):
    return "response"

# Connect the receiver
signal.connect(receiver, weak=False)

# Try to send with sender=None
print("Attempting to send signal with sender=None...")
responses = signal.send(sender=None)
print(f"Responses: {responses}")