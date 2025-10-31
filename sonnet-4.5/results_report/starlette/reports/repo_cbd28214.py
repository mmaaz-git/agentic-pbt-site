from starlette.middleware import Middleware

class TestMiddleware:
    pass

TestMiddleware.__name__ = ""

m = Middleware(TestMiddleware, 0)
print(repr(m))