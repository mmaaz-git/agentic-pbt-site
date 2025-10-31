#!/usr/bin/env python3
"""Check how email.Message behaves with deletion during iteration."""

from email.message import Message

# Test 1: Delete early key
print("Test 1: Deleting early key during iteration")
headers = Message()
headers['A_Header'] = 'a'
headers['B_Header'] = 'b'
headers['C_Header'] = 'c'
headers['D-Normal'] = 'd'

print(f"Before: {list(headers.keys())}")
deleted = []
seen = []
for k in headers:
    seen.append(k)
    print(f"  Iterating: {k}, has underscore: {'_' in k}")
    if '_' in k:
        del headers[k]
        deleted.append(k)
        print(f"    Deleted {k}")
print(f"Seen during iteration: {seen}")
print(f"Actually deleted: {deleted}")
print(f"After: {list(headers.keys())}")
print(f"Remaining with underscore: {[k for k in headers.keys() if '_' in k]}")

print("\n" + "=" * 60 + "\n")

# Test 2: Many headers
print("Test 2: Many headers with underscores")
headers = Message()
for i in range(10):
    headers[f'Header_{i}'] = f'value{i}'
headers['Normal-Header'] = 'normal'

print(f"Before: {len([k for k in headers.keys() if '_' in k])} headers with underscores")
deleted = []
for k in headers:
    if '_' in k:
        del headers[k]
        deleted.append(k)
print(f"Deleted {len(deleted)} headers: {deleted[:3]}...")
print(f"After: {len([k for k in headers.keys() if '_' in k])} headers with underscores")

remaining_underscore = [k for k in headers.keys() if '_' in k]
if remaining_underscore:
    print(f"BUG: Still have headers with underscores: {remaining_underscore}")

print("\n" + "=" * 60 + "\n")

# Test 3: Check iteration behavior
print("Test 3: Checking iteration behavior of email.Message")
headers = Message()
headers['First_Header'] = '1'
headers['Second_Header'] = '2'
headers['Third_Header'] = '3'

iteration_order = []
for k in headers:
    iteration_order.append(k)
    if len(iteration_order) == 1:  # Delete after seeing first key
        del headers['Second_Header']
        print(f"Deleted 'Second_Header' after seeing '{k}'")

print(f"Iteration saw keys: {iteration_order}")
print(f"Final keys: {list(headers.keys())}")

if 'Third_Header' not in iteration_order:
    print("BUG: Iterator skipped 'Third_Header' after deletion!")