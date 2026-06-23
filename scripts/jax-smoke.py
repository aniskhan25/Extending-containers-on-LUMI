#!/usr/bin/env python3
"""Smoke test for the LUMI JAX container."""

from __future__ import annotations

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
    packages = ["jax", "jaxlib", "optax", "flax"]

    failed = 0
    for pkg in packages:
        result = check_import(pkg)
        print(result)
        if result.startswith("FAIL"):
            failed += 1

    devices = jax.devices()
    print(f"jax.devices: {devices}")

    if not devices:
        print("FAIL  no JAX devices found")
        failed += 1
    else:
        # Run a minimal compute operation to confirm the backend works
        a = jnp.ones((4, 4))
        b = jnp.dot(a, a)
        print(f"jnp.dot result shape: {b.shape}, sum: {float(b.sum()):.1f}")

    if failed:
        raise SystemExit(f"{failed} check(s) failed")


if __name__ == "__main__":
    main()
