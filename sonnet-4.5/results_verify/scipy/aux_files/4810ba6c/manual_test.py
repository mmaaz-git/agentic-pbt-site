import scipy.datasets

print("Testing with cache directory existing...")

print("\n1. Testing with invalid string:")
try:
    scipy.datasets.clear_cache("invalid_string")
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n2. Testing with integer:")
try:
    scipy.datasets.clear_cache(42)
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n3. Testing with dict:")
try:
    scipy.datasets.clear_cache({"key": "value"})
    print("   No error raised - BUG!")
except (ValueError, TypeError, AssertionError) as e:
    print(f"   Error raised as expected: {type(e).__name__}: {e}")

print("\n4. Testing with valid None:")
try:
    scipy.datasets.clear_cache(None)
    print("   Successfully cleared cache with None")
except Exception as e:
    print(f"   Unexpected error with None: {e}")