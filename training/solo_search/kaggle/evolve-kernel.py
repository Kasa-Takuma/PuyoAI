import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.82")

input_root = Path("/kaggle/input")
markers = sorted(input_root.glob("**/solo_search/evolve_profiles.py"))
if not markers:
    flat_markers = sorted(input_root.glob("**/evolve_profiles.py"))
    if flat_markers:
        source_dir = flat_markers[-1].parent
        package_dir = Path("/kaggle/working/v12ac-evolution-source/solo_search")
        package_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.py"):
            shutil.copy2(source, package_dir / source.name)
        markers = [package_dir / "evolve_profiles.py"]
if not markers:
    archives = sorted(input_root.glob("**/solo_search.zip"))
    if archives:
        extracted_root = Path("/kaggle/working/v12ac-evolution-source")
        shutil.unpack_archive(archives[-1], extracted_root)
        markers = sorted(extracted_root.glob("**/solo_search/evolve_profiles.py"))
if not markers:
    raise FileNotFoundError("A dataset containing solo_search/evolve_profiles.py is required")

package_root = markers[-1].parent.parent
sys.path.insert(0, str(package_root))
output_dir = Path("/kaggle/working/v12ac-evolution")
output_dir.mkdir(parents=True, exist_ok=True)

from solo_search.verify_profile_parity import main as verify_profile_parity

verify_profile_parity()

resume_reports = sorted(input_root.glob("**/report.json"))
arguments = [
    "evolve_profiles",
    "--generations", "4",
    "--population", "36",
    "--stage-turns", "3000,6000,12000",
    "--stage-keeps", "12,4",
    "--stage-games", "2,3,4",
    "--beam-width", "24",
    "--profile-batch", "auto",
    "--output", str(output_dir / "report.json"),
]
if resume_reports:
    resume_path = output_dir / "resume-report.json"
    shutil.copy2(resume_reports[-1], resume_path)
    arguments.extend(["--resume-report", str(resume_path)])

sys.argv = arguments
from solo_search.evolve_profiles import main

main()
