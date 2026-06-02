#!/usr/bin/env python
"""Smoke test for the LUMI Matbench Discovery container."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from ase import Atoms
from ase.calculators.emt import EMT
from pymatgen.core import Structure


def optional_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - smoke test should report import reason
        return f"{name}: unavailable ({exc.__class__.__name__}: {exc})"
    version = getattr(module, "__version__", "unknown")
    return f"{name}: {version}"


def main() -> None:
    matbench = importlib.import_module("matbench_discovery")

    print(f"matbench_discovery: {getattr(matbench, '__version__', 'unknown')}")
    print(f"numpy: {np.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        print(f"torch.cuda.device_name.0: {torch.cuda.get_device_name(0)}")

    silicon = Structure.from_spacegroup(
        "Fd-3m",
        lattice=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
        species=["Si"],
        coords=[[0, 0, 0]],
    )
    print(f"pymatgen_formula: {silicon.composition.reduced_formula}")

    atoms = Atoms(
        "Cu2",
        positions=[[0, 0, 0], [0, 0, 2.2]],
        cell=[6, 6, 6],
        pbc=True,
        calculator=EMT(),
    )
    print(f"ase_emt_energy_ev: {atoms.get_potential_energy():.6f}")
    print(f"ase_emt_force_norm: {np.linalg.norm(atoms.get_forces()):.6f}")

    print(optional_import("mace"))
    print(optional_import("chgnet"))


if __name__ == "__main__":
    main()
