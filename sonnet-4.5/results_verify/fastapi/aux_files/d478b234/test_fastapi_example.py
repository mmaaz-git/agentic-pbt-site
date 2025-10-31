from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from fastapi.testclient import TestClient

app = FastAPI()
security = HTTPBearer()

@app.get("/protected")
def protected_route(credentials = Depends(security)):
    return {"credentials": credentials.credentials}

client = TestClient(app)

# Test with leading whitespace
response = client.get("/protected", headers={"Authorization": " Bearer validtoken"})
print(f"With leading space:")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

# Test without leading whitespace (should work)
response = client.get("/protected", headers={"Authorization": "Bearer validtoken"})
print(f"\nWithout leading space:")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")