from Cython.Build.Dependencies import DistutilsInfo

info1 = DistutilsInfo()
info2 = DistutilsInfo()
info2.values['libraries'] = ['lib1', 'lib2']

result = info1.merge(info2)

print(f"Same object: {result.values['libraries'] is info2.values['libraries']}")

result.values['libraries'].append('lib3')

print(f"info2 libraries: {info2.values['libraries']}")
print(f"result libraries: {result.values['libraries']}")

assert result.values['libraries'] is not info2.values['libraries'], \
    "Lists should be independent"