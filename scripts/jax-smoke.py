#!/usr/bin/env python3
"""Smoke test for the LUMI JAX container.

Run during build (%test): validates imports only — no GPU required.
Run post-build on a GPU node to validate the ROCm backend:
    singularity exec --rocm container.sif python3 /opt/jax-smoke.py --gpu
"""

from __future__ import annotations

import argparse
import importlib

import jax
import jax.numpy as jnp


def check_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL  {name}: {exc.__class__.__name__}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return f"ok    {name}: {version}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Assert GPU backend (run on compute node)")
    args = parser.parse_args()

    packages = ["jax", "jaxlib", "optax", "flax"]

    failed = 0
    for pkg in packages:
        result = check_import(pkg)
        print(result)
        if result.startswith("FAIL"):
            failed += 1

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"jax.default_backend: {backend}")
    print(f"jax.devices: {devices}")

    if args.gpu:
        if backend != "gpu":
            print(f"FAIL  expected gpu backend, got {backend}")
            failed += 1
        elif not devices:
            print("FAIL  no JAX devices found")
            failed += 1

    # Basic compute — runs on whatever backend is available
    a = jnp.ones((4, 4))
    b = jnp.dot(a, a)
    print(f"jnp.dot result shape: {b.shape}, sum: {float(b.sum()):.1f}")

    if failed:
        raise SystemExit(f"{failed} check(s) failed")


if __name__ == "__main__":
    main()
