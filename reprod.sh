set -euxo pipefail

git clone https://github.com/yt-project/yt --depth=1 _yt

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e "./_yt[test]"
python -m pip install memray
python -m pip install setuptools # ...

python -m memray run t.py

#cd _yt/
#python -m memray run -m pytest yt/data_objects

# ...
