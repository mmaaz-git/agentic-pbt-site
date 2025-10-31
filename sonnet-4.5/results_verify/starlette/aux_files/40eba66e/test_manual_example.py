from starlette.middleware import Middleware


class CallableWithoutName:
    def __call__(self, app, *args, **kwargs):
        return app


callable_obj = CallableWithoutName()
middleware = Middleware(callable_obj, 123, kwarg="test")

print(f"Repr output: {repr(middleware)}")
print(f"Expected: Middleware(CallableWithoutName, 123, kwarg='test') or Middleware(123, kwarg='test')")

# Test with lambda (which also doesn't have __name__ in some cases)
lambda_middleware = Middleware(lambda app: app, 456, another="value")
print(f"\nLambda repr: {repr(lambda_middleware)}")

# Test with no args
no_args_middleware = Middleware(callable_obj)
print(f"\nNo args repr: {repr(no_args_middleware)}")

# Test with only kwargs
kwargs_only_middleware = Middleware(callable_obj, key="value", another="test")
print(f"\nKwargs only repr: {repr(kwargs_only_middleware)}")