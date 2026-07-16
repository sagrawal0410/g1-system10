#!/usr/bin/env python3
"""Standalone Phase D test runner (no pytest dependency).

Run from the scripts/ dir (so `phase_d` is importable):
    python -m phase_d.tests.run_all
Exits non-zero if any test fails.
"""
from __future__ import annotations

import importlib
import traceback

TEST_MODULES = [
    "phase_d.tests.test_layout",
    "phase_d.tests.test_fsq",
    "phase_d.tests.test_ensembler",
    "phase_d.tests.test_stitching",
    "phase_d.tests.test_reranker",
    "phase_d.tests.test_wrappers",
    "phase_d.tests.test_decoder_onnx",
]


def main() -> int:
    passed = failed = 0
    failures = []
    for modname in TEST_MODULES:
        mod = importlib.import_module(modname)
        tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
        print(f"\n=== {modname} ({len(tests)} tests) ===")
        for fn in tests:
            try:
                fn()
                print(f"  PASS {fn.__name__}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {fn.__name__}: {e}")
                traceback.print_exc()
                failures.append(f"{modname}.{fn.__name__}: {e}")
                failed += 1
    print(f"\n==== {passed} passed, {failed} failed ====")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
