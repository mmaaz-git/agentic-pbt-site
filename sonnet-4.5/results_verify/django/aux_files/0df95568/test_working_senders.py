from django.dispatch import Signal

class MyClass:
    pass

def test_sender(sender, description, use_caching):
    signal = Signal(use_caching=True)
    
    def receiver(**kwargs):
        return 42
    
    signal.connect(receiver, sender=sender, weak=False)
    
    try:
        result = signal.has_listeners(sender=sender)
        print(f"✓ {description} with use_caching={use_caching}: {result}")
    except TypeError as e:
        print(f"✗ {description} with use_caching={use_caching}: TypeError - {e}")

# Test sender types that should work
print("Testing sender types that support weak references:")
print("="*50)

test_cases = [
    (MyClass(), "custom class instance"),
    (MyClass, "custom class"),
    (lambda: None, "lambda"),
    (test_sender, "function"),
]

for sender, description in test_cases:
    test_sender(sender, description, use_caching=True)
