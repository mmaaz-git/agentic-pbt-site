from hypothesis import given, strategies as st, settings
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient


class UncopyableClass:
    def __init__(self, value: int = 42):
        self.value = value

    def __deepcopy__(self, memo):
        raise TypeError("Cannot deepcopy this object")


@given(st.integers())
@settings(max_examples=10)
def test_uncopyable_defaults_should_not_crash(value):
    app = FastAPI()
    uncopyable = UncopyableClass(value)

    @app.get("/test")
    def endpoint(param: int = Query(default=uncopyable)):
        return {"param": param}

    client = TestClient(app)
    try:
        response = client.get("/test")
        assert response.status_code == 200
        print(f"Test with value={value}: PASSED (status={response.status_code})")
    except Exception as e:
        print(f"Test with value={value}: FAILED with {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    test_uncopyable_defaults_should_not_crash()