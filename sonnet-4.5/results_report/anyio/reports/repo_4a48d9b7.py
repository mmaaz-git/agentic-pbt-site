import asyncio
import math

import anyio
from anyio import fail_after


async def main():
    with fail_after(math.nan):
        await anyio.sleep(0.01)


asyncio.run(main())