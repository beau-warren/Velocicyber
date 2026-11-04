"""Exec a local CPU server that exits if its owning harness disappears."""

import ctypes
import os
import signal
import sys

parent = int(sys.argv[1])
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG on Linux
    raise OSError(ctypes.get_errno(), "Could not bind CPU server lifetime to harness")
if os.getppid() != parent:
    raise SystemExit("Harness exited before CPU server startup")
os.execv(sys.argv[2], sys.argv[2:])
