#!/usr/bin/env python
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poibench.evaluate import main

if __name__ == "__main__":
    main()
