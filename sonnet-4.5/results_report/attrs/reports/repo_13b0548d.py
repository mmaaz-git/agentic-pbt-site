from attrs import cmp_using

try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(e)