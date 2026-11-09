import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "tmp_tests")
os.makedirs(TMP, exist_ok=True)
tempfile.tempdir = TMP
