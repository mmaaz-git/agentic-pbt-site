import asyncio
import base64
from fastapi.security import HTTPBasic
from starlette.requests import Request
from starlette.datastructures import Headers


async def test_utf8_password():
    username = "admin"
    password = "Pässwörd123"

    credentials_str = f"{username}:{password}"
    encoded = base64.b64encode(credentials_str.encode('utf-8')).decode('ascii')
    auth_header = f"Basic {encoded}"

    request = Request({
        'type': 'http',
        'headers': Headers({'authorization': auth_header}).raw,
    })

    security = HTTPBasic()
    try:
        result = await security(request)
        print(f"Username: {result.username}, Password: {result.password}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}")
        return False
    return True


if __name__ == "__main__":
    success = asyncio.run(test_utf8_password())
    print(f"Test passed: {success}")