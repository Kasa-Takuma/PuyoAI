import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.42")

input_root = Path("/kaggle/input")
package_markers = sorted(input_root.glob("**/solo_search/__init__.py"))
trajectory_files = sorted(input_root.glob("**/initial-v13-trajectories.jsonl*"))
seed_files = sorted(input_root.glob("**/dev-pairs.json"))
if not package_markers:
    raise FileNotFoundError("solo_search package was not found in the attached Kaggle datasets")
if not trajectory_files:
    raise FileNotFoundError("initial-v13-trajectories.jsonl was not found in the attached Kaggle datasets")
if not seed_files:
    raise FileNotFoundError("dev-pairs.json was not found in the attached Kaggle datasets")

package_root = package_markers[-1].parent.parent
sys.path.insert(0, str(package_root))
os.chdir(package_root)

from solo_search.train import main as train_main

sys.argv = [
    "train.py",
    "--initial-data",
    str(trajectory_files[-1]),
    "--output-dir",
    "/kaggle/working/solo-search/model",
    "--name",
    "solo_value_initial",
    "--epochs",
    "80",
    "--batch-size",
    "1024",
]
train_main()

from solo_search.simulate import run_simulation

report = run_simulation(
    seed_files[-1],
    "/kaggle/working/solo-search/model/solo_value.web.json",
    "/kaggle/working/solo-search/simulation",
    max_turns=1000,
    beam_width=32,
)
print(report)
