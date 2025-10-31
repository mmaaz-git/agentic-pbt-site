name = 'foo"bar'

def django_quote_name(name):
    if name.startswith('"') and name.endswith('"'):
        return name
    return '"%s"' % name

def is_properly_quoted(name, quoted):
    if not quoted.startswith('"') or not quoted.endswith('"'):
        return False
    inner = quoted[1:-1]
    unescaped = inner.replace('""', '"')
    return unescaped == name

quoted = django_quote_name(name)
print(f'name: {name!r}')
print(f'quoted: {quoted!r}')
print(f'inner: {quoted[1:-1]!r}')
unescaped = quoted[1:-1].replace('""', '"')
print(f'unescaped: {unescaped!r}')
print(f'unescaped == name: {unescaped == name}')
print(f'is_properly_quoted: {is_properly_quoted(name, quoted)}')