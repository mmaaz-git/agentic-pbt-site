from pandas.io.formats.printing import _justify

# Test case demonstrating the bug where _justify silently drops elements
head = [['a', 'b', 'c']]
tail = [['x']]
result_head, result_tail = _justify(head, tail)

print(f"Input:  head={head}, tail={tail}")
print(f"Output: head={result_head}, tail={result_tail}")
print(f"Expected: head should have 3 elements, tail should have 1 element")
print(f"Actual:   head has {len(result_head[0])} element(s) (lost 2!), tail has {len(result_tail[0])} element(s)")
print()
print("Data Loss: Elements 'b' and 'c' were silently dropped from head!")