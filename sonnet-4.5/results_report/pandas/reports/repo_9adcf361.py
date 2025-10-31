from Cython.Plex.Actions import Call
import functools

# Test Case 1: Callable object (class with __call__ method)
class CallableObject:
    def __call__(self, scanner, text):
        return 'result'

print("Test 1: Callable object")
try:
    action1 = Call(CallableObject())
    print(repr(action1))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 2: functools.partial
def base_func(scanner, text, extra):
    return extra

print("\nTest 2: functools.partial")
try:
    action2 = Call(functools.partial(base_func, extra=10))
    print(repr(action2))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 3: Lambda function (should work)
print("\nTest 3: Lambda function")
try:
    action3 = Call(lambda scanner, text: 'lambda_result')
    print(repr(action3))
except AttributeError as e:
    print(f"AttributeError: {e}")

# Test Case 4: Regular function (should work)
def regular_func(scanner, text):
    return 'regular_result'

print("\nTest 4: Regular function")
try:
    action4 = Call(regular_func)
    print(repr(action4))
except AttributeError as e:
    print(f"AttributeError: {e}")