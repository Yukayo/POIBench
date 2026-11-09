#!/usr/bin/env python
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tests

if __name__ == "__main__":
    mods = [a for a in sys.argv[1:] if not a.startswith("-")]
    if mods:
        suite = unittest.defaultTestLoader.loadTestsFromNames([f"tests.{m}" for m in mods])
    else:
        suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "tests"), pattern="test_*.py", top_level_dir=ROOT)
    res = unittest.TextTestRunner(verbosity=2 if "-v" in sys.argv else 1).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
