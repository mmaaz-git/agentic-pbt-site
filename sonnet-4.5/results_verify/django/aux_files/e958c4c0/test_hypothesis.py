import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st
from django.conf import Settings


def test_timezone_path_should_stay_within_zoneinfo_root():
    zoneinfo_root = Path("/usr/share/zoneinfo")

    if not zoneinfo_root.exists():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        external_file = Path(tmpdir) / "fake_timezone"
        external_file.touch()

        relative_path = os.path.relpath(external_file, zoneinfo_root)

        parts = relative_path.split("/")
        zone_info_file = zoneinfo_root.joinpath(*parts)

        is_inside = (
            zoneinfo_root.resolve() == zone_info_file.resolve() or
            zoneinfo_root.resolve() in zone_info_file.resolve().parents
        )

        if zone_info_file.exists() and not is_inside:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"TIME_ZONE = {repr(relative_path)}\n")
                f.write("SECRET_KEY = 'test'\n")
                settings_file = f.name

            try:
                module_name = Path(settings_file).stem
                sys.path.insert(0, str(Path(settings_file).parent))

                try:
                    settings_obj = Settings(module_name)
                    assert False, "Django accepted timezone path outside zoneinfo_root"
                except ValueError:
                    pass
                finally:
                    sys.path.remove(str(Path(settings_file).parent))
                    if module_name in sys.modules:
                        del sys.modules[module_name]
            finally:
                os.unlink(settings_file)


# Run the test
test_timezone_path_should_stay_within_zoneinfo_root()
print("Test completed")