from starlette.responses import FileResponse

file_size = 1000
http_range = "bytes=100-199,400-499,150-450"

result = FileResponse._parse_range_header(http_range, file_size)

print(f"Input: http_range = {repr(http_range)}, file_size = {file_size}")
print(f"Result: {result}")
print()

for i in range(len(result) - 1):
    start1, end1 = result[i]
    start2, end2 = result[i+1]
    if end1 > start2:
        print(f"BUG: Ranges overlap: ({start1}, {end1}) and ({start2}, {end2})")
        print(f"  Range {i} ends at {end1}, but range {i+1} starts at {start2}")
        print(f"  These ranges should have been merged into a single range")

if len(result) == 2 and result[0][0] == 100 and result[0][1] == 451 and result[1][0] == 400 and result[1][1] == 500:
    print("\nExpected behavior: These overlapping ranges should be merged into [(100, 500)]")