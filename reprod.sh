set -euxo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pip install memray
python -m memray run t.py
