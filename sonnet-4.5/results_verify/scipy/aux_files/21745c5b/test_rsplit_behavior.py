import pandas as pd

# Test how rsplit currently handles patterns
s = pd.Series(['a.b.c.d', 'a+b+c+d', 'a.*b.*c.*d'])

print("Testing rsplit with literal dot:")
print(s.str.rsplit('.'))

print("\nTesting rsplit with plus sign:")
print(s.str.rsplit('+'))

print("\nTesting rsplit with regex pattern '.*':")
print(s.str.rsplit('.*'))

print("\nTesting split with regex=True on '.*':")
print(s.str.split('.*', regex=True))

print("\nTesting split with regex=False on '.*':")
print(s.str.split('.*', regex=False))

# Check if rsplit treats pattern as literal or regex by default
s2 = pd.Series(['foo|bar|baz'])
print("\nTesting rsplit with pipe character '|':")
print(s2.str.rsplit('|'))

print("\nTesting split with pipe character '|' and regex=True:")
print(s2.str.split('|', regex=True))

print("\nTesting split with pipe character '|' and regex=False:")
print(s2.str.split('|', regex=False))