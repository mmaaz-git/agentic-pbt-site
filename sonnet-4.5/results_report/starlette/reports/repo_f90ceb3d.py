from starlette.middleware import Middleware


class CallableFactory:
    def __call__(self, app):
        return app


factory = CallableFactory()
middleware = Middleware(factory, "arg1", "arg2", key="value")

print(repr(middleware))