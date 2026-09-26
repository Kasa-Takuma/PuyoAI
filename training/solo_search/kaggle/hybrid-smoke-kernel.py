import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.42")

input_root = Path("/kaggle/input")
package_markers = sorted(input_root.glob("**/solo_search/__init__.py"))
model_files = sorted(input_root.glob("**/solo_value.web.json"))
seed_files = sorted(input_root.glob("**/test-pairs.json"))
if not package_markers:
    raise FileNotFoundError("solo_search package was not found in the attached Kaggle datasets")
if not model_files:
    raise FileNotFoundError("solo_value.web.json was not found in the attached Kaggle datasets")
if not seed_files:
    raise FileNotFoundError("test-pairs.json was not found in the attached Kaggle datasets")

package_root = package_markers[-1].parent.parent
sys.path.insert(0, str(package_root))
os.chdir(package_root)

from solo_search.simulate import run_simulation

report = run_simulation(
    seed_files[-1],
    model_files[-1],
    "/kaggle/working/solo-search/hybrid-smoke",
    max_turns=16,
    beam_width=32,
    limit_games=32,
)
print(report)
