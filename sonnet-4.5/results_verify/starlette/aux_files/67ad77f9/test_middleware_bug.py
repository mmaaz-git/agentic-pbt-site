"""Test script to reproduce the Middleware.__repr__ bug."""

from starlette.middleware import Middleware


# Test case 1: Reproduce the exact bug with callable instance
class CallableFactory:
    def __call__(self, app):
        return app


factory = CallableFactory()
middleware = Middleware(factory, "arg1", "arg2", key="value")

print("Test 1 - Callable instance without __name__:")
print(f"  repr output: {repr(middleware)}")
print(f"  Starts with 'Middleware(,'? {repr(middleware).startswith('Middleware(,')}")
print()

# Test case 2: Normal function (has __name__)
def normal_factory(app):
    return app

middleware2 = Middleware(normal_factory, "arg1", "arg2", key="value")
print("Test 2 - Normal function with __name__:")
print(f"  repr output: {repr(middleware2)}")
print()

# Test case 3: Class with __name__ attribute
class FactoryWithName:
    __name__ = "CustomFactory"
    def __call__(self, app):
        return app

factory3 = FactoryWithName()
middleware3 = Middleware(factory3, "arg1", "arg2", key="value")
print("Test 3 - Callable instance with __name__ attribute:")
print(f"  repr output: {repr(middleware3)}")
print()

# Test case 4: Empty args and kwargs with callable instance
factory4 = CallableFactory()
middleware4 = Middleware(factory4)
print("Test 4 - Callable instance with no args/kwargs:")
print(f"  repr output: {repr(middleware4)}")
print(f"  Is it just 'Middleware()'? {repr(middleware4) == 'Middleware()'}")