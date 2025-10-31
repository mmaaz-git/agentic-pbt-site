from django.core.files.base import File

class FalsyButOpenFile:
    def __bool__(self):
        return False

    @property
    def closed(self):
        return False

falsy_file = FalsyButOpenFile()
file_obj = File(falsy_file)

print(f"Underlying file closed: {falsy_file.closed}")
print(f"FileProxyMixin closed: {file_obj.closed}")

assert file_obj.closed == False, f"Expected False, got {file_obj.closed}"