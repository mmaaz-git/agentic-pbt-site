import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

# Manually trace through the logic for size=0, chunk_size=5, region=None
size = 0
chunk_size = 5
region = None

# Line 141-142
if region is None:
    region = slice(0, size)
print(f"region after line 142: {region}")

# Line 144
region_start = region.start or 0  
print(f"region_start: {region_start}")

# Line 146
chunks_on_region = [chunk_size - (region_start % chunk_size)]
print(f"Initial chunks_on_region: {chunks_on_region}")
print(f"chunks_on_region[0] = {chunk_size} - ({region_start} % {chunk_size}) = {chunks_on_region[0]}")

# Line 147 - extend
num_full_chunks = (size - chunks_on_region[0]) // chunk_size
print(f"(size - chunks_on_region[0]) // chunk_size = ({size} - {chunks_on_region[0]}) // {chunk_size} = {num_full_chunks}")
chunks_on_region.extend([chunk_size] * num_full_chunks)
print(f"After extend: {chunks_on_region}")

# Line 148-149 - check for remainder
remainder = (size - chunks_on_region[0]) % chunk_size
print(f"(size - chunks_on_region[0]) % chunk_size = ({size} - {chunks_on_region[0]}) % {chunk_size} = {remainder}")
if remainder != 0:
    chunks_on_region.append(remainder)
    print(f"After append remainder: {chunks_on_region}")
else:
    print(f"No remainder to append")

# Line 150 - return
result = tuple(chunks_on_region)
print(f"Final result: {result}")
print(f"Sum of result: {sum(result)}")
print(f"Expected sum: {size}")
print(f"BUG: {sum(result)} != {size}")
