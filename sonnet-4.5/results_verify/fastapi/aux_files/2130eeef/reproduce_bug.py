#!/usr/bin/env python3
"""Script to reproduce the Middleware __repr__ bug"""

from fastapi.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


# Test case 1: Simple reproduction
print("=== Test 1: Simple reproduction ===")
middleware = Middleware(CallableWithoutName(), 123, foo="bar")
repr_str = repr(middleware)
print(f"repr output: {repr_str}")
print(f"Has leading comma? {repr_str.startswith('Middleware(,')}")

# Test case 2: With no args
print("\n=== Test 2: No args ===")
middleware2 = Middleware(CallableWithoutName())
repr_str2 = repr(middleware2)
print(f"repr output: {repr_str2}")
print(f"Has leading comma? {repr_str2.startswith('Middleware(,')}")

# Test case 3: With regular function (has __name__)
print("\n=== Test 3: Regular function with __name__ ===")
def my_middleware(app, *args, **kwargs):
    return app

middleware3 = Middleware(my_middleware, 456, bar="baz")
repr_str3 = repr(middleware3)
print(f"repr output: {repr_str3}")
print(f"Has leading comma? {repr_str3.startswith('Middleware(,')}")

# Test case 4: Lambda function
print("\n=== Test 4: Lambda function ===")
middleware4 = Middleware(lambda app: app, 789)
repr_str4 = repr(middleware4)
print(f"repr output: {repr_str4}")
print(f"Has leading comma? {repr_str4.startswith('Middleware(,')}")