from starlette.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


# Create an instance of the callable without __name__ attribute
callable_obj = CallableWithoutName()

# Create Middleware with this callable and some arguments
middleware = Middleware(callable_obj, 123, kwarg="test")

# Print the repr to demonstrate the bug
print(repr(middleware))