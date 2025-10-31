#!/usr/bin/env python3
"""Minimal reproduction of the Middleware __repr__ bug with leading comma."""

from fastapi.middleware import Middleware


class CallableWithoutName:
    """A callable class that doesn't have a __name__ attribute."""
    def __call__(self, app, *args, **kwargs):
        return app


# Create a Middleware instance with a callable lacking __name__
# and some additional arguments
middleware = Middleware(CallableWithoutName(), 123, foo="bar")

# Print the repr output
print("repr(middleware) output:")
print(repr(middleware))

# Also test without any args
middleware_no_args = Middleware(CallableWithoutName())
print("\nrepr(middleware_no_args) output:")
print(repr(middleware_no_args))

# Compare with a regular function that has __name__
def my_middleware(app):
    return app

middleware_with_name = Middleware(my_middleware, 456, bar="baz")
print("\nrepr(middleware_with_name) output:")
print(repr(middleware_with_name))

# Also test with a lambda (which has __name__ = '<lambda>')
middleware_lambda = Middleware(lambda app: app, 789)
print("\nrepr(middleware_lambda) output:")
print(repr(middleware_lambda))