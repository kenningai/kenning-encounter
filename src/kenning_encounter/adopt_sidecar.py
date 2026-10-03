"""python -m kenning_encounter.adopt_sidecar --adopt PROVIDER:MODEL [--sidecar PATH]

Adopt a pre-v0.17.0 meaning sidecar for the voice that wrote it. A module of
its own because the package imports `meaning` on load, and running a module
already imported warns at the operator for nothing.
"""

from .meaning import _main

if __name__ == "__main__":
    raise SystemExit(_main())
