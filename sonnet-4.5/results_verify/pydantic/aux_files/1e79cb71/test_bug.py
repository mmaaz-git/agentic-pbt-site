import anyio
from anyio.streams.memory import (
    MemoryObjectSendStream,
    MemoryObjectReceiveStream,
    MemoryObjectStreamState,
)
from anyio import WouldBlock


async def main():
    # Test 1: Try using create_memory_object_stream with fractional buffer
    print("Test 1: Using create_memory_object_stream with fractional buffer")
    try:
        from anyio import create_memory_object_stream
        send_stream, receive_stream = create_memory_object_stream(max_buffer_size=10.5)
        print("  Created successfully with max_buffer_size=10.5 via create_memory_object_stream")
    except ValueError as e:
        print(f"  ValueError: {e}")

    print("\nTest 2: Using MemoryObjectStreamState directly with fractional buffer")
    # The bug report tests this by directly using MemoryObjectStreamState
    state = MemoryObjectStreamState[int](max_buffer_size=10.5)
    send_stream = MemoryObjectSendStream(state)
    receive_stream = MemoryObjectReceiveStream(state)

    # Try to add 11 items (should only allow 10 if floor is used)
    for i in range(11):
        send_stream.send_nowait(i)

    stats = send_stream.statistics()
    print(f"  max_buffer_size: {stats.max_buffer_size}")
    print(f"  current_buffer_used: {stats.current_buffer_used}")

    # Try to add 12th item
    try:
        send_stream.send_nowait(11)
        print(f"  Successfully added 12th item (total: {stats.current_buffer_used + 1})")
    except WouldBlock:
        print(f"  Raised WouldBlock after {stats.current_buffer_used} items")
        print(f"  Expected by bug report: floor(10.5) = 10 items")
        print(f"  Actual: ceil(10.5) = 11 items")

    send_stream.close()
    receive_stream.close()

    print("\nTest 3: Testing with buffer_size=1.5")
    state = MemoryObjectStreamState[int](max_buffer_size=1.5)
    send_stream = MemoryObjectSendStream(state)
    receive_stream = MemoryObjectReceiveStream(state)

    # Try to add items
    send_stream.send_nowait(0)
    print("  Added 1st item")

    try:
        send_stream.send_nowait(1)
        print("  Added 2nd item - buffer accepts more than floor(1.5)=1")
    except WouldBlock:
        print("  WouldBlock after 1 item")

    send_stream.close()
    receive_stream.close()


anyio.run(main)