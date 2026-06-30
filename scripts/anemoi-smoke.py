#!/usr/bin/env python
"""Smoke test for the LUMI Anemoi container."""

from __future__ import annotations

import importlib

import torch


def check_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL  {name}: {exc.__class__.__name__}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return f"ok    {name}: {version}"


def main() -> None:
    packages = [
        "anemoi.training",
        "anemoi.models",
        "anemoi.graphs",
        "zarr",
        "trimesh",
        "pyshtools",
    ]

    print(f"torch: {torch.__version__}")
    print(f"torch.version.hip: {torch.version.hip}")
    assert torch.version.hip is not None, (
        f"torch is not a ROCm build (hip={torch.version.hip}); "
        "a CPU/CUDA wheel was likely pulled in during install"
    )
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        print(f"torch.cuda.device_name.0: {torch.cuda.get_device_name(0)}")

    failed = 0
    for pkg in packages:
        result = check_import(pkg)
        print(result)
        if result.startswith("FAIL"):
            failed += 1

    if failed:
        raise SystemExit(f"{failed} import(s) failed")


if __name__ == "__main__":
    main()
