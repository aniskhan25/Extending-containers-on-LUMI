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
    path = getattr(module, "__file__", "unknown")
    return f"ok    {name}: {version}  ({path})"


def version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_below(name: str, exclusive_max: str) -> None:
    """Guard against the LAIF base's /opt/venv shadowing this venv's pins.

    The base image puts its own site-packages on PYTHONPATH ahead of
    /opt/anemoi-venv, so a package pinned here can be silently overridden by a
    newer copy from the base. Only the version actually imported matters.
    """
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "0")
    assert version_tuple(version) < version_tuple(exclusive_max), (
        f"{name} {version} resolved from {module.__file__}; anemoi requires "
        f"{name}<{exclusive_max}. The base image's copy is shadowing "
        "/opt/anemoi-venv (check PYTHONPATH ordering in %environment)"
    )


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

    # anemoi-datasets requires zarr<3 and numcodecs<0.16; both exist in the
    # base image at newer versions, so assert the pinned ones win.
    check_below("zarr", "3")
    check_below("numcodecs", "0.16")


if __name__ == "__main__":
    main()
