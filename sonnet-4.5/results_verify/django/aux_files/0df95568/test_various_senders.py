from django.dispatch import Signal

def test_sender(sender, description, use_caching):
    signal = Signal(use_caching=use_caching)
    
    def receiver(**kwargs):
        return 42
    
    signal.connect(receiver, sender=sender, weak=False)
    
    try:
        result = signal.has_listeners(sender=sender)
        print(f"✓ {description} with use_caching={use_caching}: {result}")
    except TypeError as e:
        print(f"✗ {description} with use_caching={use_caching}: TypeError - {e}")

# Test various sender types
print("Testing various sender types:")
print("="*50)

test_cases = [
    (object(), "object()"),
    (42, "int (42)"),
    ("string", "string"),
    ((1, 2, 3), "tuple"),
    (None, "None"),
    ([1, 2, 3], "list"),
    ({"a": 1}, "dict"),
]

for sender, description in test_cases:
    test_sender(sender, description, use_caching=False)
    test_sender(sender, description, use_caching=True)
    print()
