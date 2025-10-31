from fastapi import FastAPI, Query
from fastapi.testclient import TestClient


class UncopyableDefault:
    def __init__(self, value: int = 42):
        self.value = value

    def __deepcopy__(self, memo):
        raise TypeError("Cannot deepcopy this object")


app = FastAPI()
uncopyable = UncopyableDefault()


@app.get("/test")
def endpoint(param: int = Query(default=uncopyable)):
    return {"param": param}


client = TestClient(app)
print("Making request to /test endpoint...")
try:
    response = client.get("/test")
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.json()}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()