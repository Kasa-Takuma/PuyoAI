import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.42")

input_root = Path("/kaggle/input")
package_markers = sorted(input_root.glob("**/solo_search/__init__.py"))
fixtures = sorted(input_root.glob("**/jax-fixtures.jsonl*"))
if not package_markers:
    raise FileNotFoundError("solo_search package was not found in the attached Kaggle datasets")
if not fixtures:
    raise FileNotFoundError("jax-fixtures.jsonl was not found in the attached Kaggle datasets")

package_root = package_markers[-1].parent.parent
sys.path.insert(0, str(package_root))
sys.argv.extend(["--fixtures", str(fixtures[-1])])
os.chdir(package_root)

from solo_search.kaggle_runner import main

main()
