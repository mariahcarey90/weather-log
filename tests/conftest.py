import pathlib, sys

# let tests import the repo's modules (score, fetch) wherever pytest is started from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
