#!/usr/bin/env python3
"""Smoke test for the LUMI TransformerEngine container.

Run during build (%test): validates the ROCm torch stack, that TE's flash-attn
version gate now accepts the installed flash-attn, and that TE actually bound the
flash-attn entry points. No GPU required — TE's gate runs on import.

Run post-build on a GPU node to validate which backend a real forward+backward
lands on:
    singularity exec container.sif python3 /opt/transformer-engine-smoke.py --gpu

The --gpu check runs DotProductAttention twice, once normally and once with
NVTE_FLASH_ATTN=0, and asserts the default run selects FlashAttention, uses less
memory than the unfused reference, and agrees with it numerically. Each run is a
separate process because TE evaluates NVTE_FLASH_ATTN at import time.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version as dist_version
from pathlib import Path

# Attention shape used for the backend check and the reference comparison.
BATCH, SEQ, HEADS, HEAD_DIM = 2, 4096, 16, 64

# bf16 accumulation order differs between the fused and unfused kernels.
ATOL, RTOL = 2e-2, 2e-2


def check_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return f"FAIL  {name}: {exc.__class__.__name__}: {exc}"
    return f"ok    {name}: {getattr(module, '__version__', 'unknown')}"


def flash_attn_utils():
    """Return TE's flash-attn version-bound holder.

    Named FlashAttentionUtils in TE 2.8; fall back to finding it by shape so a
    rename in a future LAIF base does not turn this test into a false pass.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import utils

    wanted = {"version", "version_required", "max_version", "is_installed"}
    holder = getattr(utils, "FlashAttentionUtils", None)
    if holder is not None and wanted <= set(vars(holder)):
        return "FlashAttentionUtils", holder
    for name in dir(utils):
        obj = getattr(utils, name)
        if wanted <= set(getattr(obj, "__dict__", {})):
            return name, obj
    raise SystemExit(
        f"could not locate TE's flash-attn version bounds in {utils.__file__} — "
        "TE's layout changed; re-run the introspection step in README.org"
    )


def check_versions() -> int:
    """CPU-safe checks. Returns the number of failures."""
    failed = 0

    for pkg in ["torch", "flash_attn", "transformer_engine", "transformer_engine_torch"]:
        result = check_import(pkg)
        print(result)
        if result.startswith("FAIL"):
            failed += 1
    if failed:
        return failed

    import torch

    print(f"torch.version.hip:    {torch.version.hip}")
    if torch.version.hip is None or "+rocm" not in torch.__version__:
        print(
            f"FAIL  torch is not a ROCm build (version={torch.__version__}, "
            f"hip={torch.version.hip}); a CPU/CUDA wheel was likely pulled in"
        )
        failed += 1

    # TE gates on distribution metadata, not flash_attn.__version__, and LAIF
    # retags the wheel — so report the string TE actually compares against.
    print(f"flash-attn metadata:  {dist_version('flash-attn')}")

    name, fa = flash_attn_utils()
    print(f"{name}: version={fa.version} accepted=[{fa.version_required}, "
          f"{fa.max_version}] is_installed={fa.is_installed}")

    if not fa.is_installed:
        print(
            f"FAIL  TE gated flash-attn off at import time: {fa.version} is outside "
            f"[{fa.version_required}, {fa.max_version}]. The FlashAttention backend "
            "cannot run, and with fused attention absent from this image TE falls "
            "back to UnfusedDotProductAttention — materialized O(seq^2) scores, "
            "which is the OOM this container exists to fix."
        )
        failed += 1

    # These stay None unless the gate above passed, so they are the ground truth
    # for whether the FlashAttention backend is usable at all.
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    for symbol in ("flash_attn_func", "flash_attn_varlen_func",
                   "_flash_attn_fwd", "_flash_attn_bwd"):
        bound = getattr(backends, symbol, None)
        if bound is None:
            print(f"FAIL  backends.{symbol} is None — TE did not bind flash-attn")
            failed += 1
        else:
            print(f"ok    backends.{symbol}")

    # The base builds flash-attn with GPU_ARCHS=gfx90a and without
    # FLASH_ATTENTION_TRITON_AMD_ENABLE, so the compiled extension must be present.
    try:
        importlib.import_module("flash_attn_2_cuda")
        print("ok    flash_attn_2_cuda (compiled gfx90a extension)")
    except ImportError as exc:
        print(f"FAIL  flash_attn_2_cuda not importable: {exc}")
        failed += 1

    return failed


def make_dpa(seq: int):
    """Build a DotProductAttention module and matching bf16 q/k/v on the GPU."""
    import torch
    import transformer_engine.pytorch as te

    qkv = [
        torch.randn(
            BATCH, seq, HEADS, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda", requires_grad=True,
        )
        for _ in range(3)
    ]
    module = te.DotProductAttention(
        num_attention_heads=HEADS,
        kv_channels=HEAD_DIM,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    ).cuda()
    return module, qkv


def selected_backend() -> str:
    """Report which backend TE chose on the most recent forward."""
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa,
    )

    backends = dpa._attention_backends
    if backends["use_flash_attention"]:
        return "FlashAttention"
    if backends["use_fused_attention"]:
        return "FusedAttention"
    return "Unfused"


def dpa_child(out_path: Path) -> None:
    """Child process: one DotProductAttention forward+backward, tensors saved."""
    import torch

    torch.manual_seed(0)
    module, qkv = make_dpa(SEQ)

    torch.cuda.reset_peak_memory_stats()
    out = module(*qkv)
    out.sum().backward()
    torch.cuda.synchronize()

    torch.save(
        {
            "out": out.detach().float().cpu(),
            **{f"grad{i}": t.grad.float().cpu() for i, t in enumerate(qkv)},
        },
        out_path,
    )
    print(json.dumps({
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "backend": selected_backend(),
    }))


def run_dpa(use_flash: bool, out_path: Path) -> tuple[str, float, str]:
    """Run one forward+backward in a child process. Returns (backend, peak MiB, log)."""
    env = dict(os.environ, NVTE_DEBUG="1", NVTE_DEBUG_LEVEL="2")
    if not use_flash:
        env["NVTE_FLASH_ATTN"] = "0"

    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--dpa-child", str(out_path)],
        env=env, capture_output=True, text=True, check=False,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(
            f"attention run failed (use_flash={use_flash}, rc={proc.returncode}):\n{log}"
        )

    report = json.loads(proc.stdout.strip().splitlines()[-1])
    return report["backend"], report["peak_mib"], log


def check_gpu() -> int:
    """Compare the fused path against a forced-unfused reference. Returns failures."""
    import torch

    if not torch.cuda.is_available():
        print("FAIL  no ROCm device visible; run inside a GPU allocation with "
              "lumi-aif-singularity-bindings loaded")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        flash_path, unfused_path = Path(tmp) / "flash.pt", Path(tmp) / "unfused.pt"
        backend, flash_peak, flash_log = run_dpa(True, flash_path)
        _, unfused_peak, _ = run_dpa(False, unfused_path)

        print(f"backend: {backend} ({flash_peak:.0f} MiB peak) vs "
              f"{unfused_peak:.0f} MiB with NVTE_FLASH_ATTN=0")

        if backend != "FlashAttention":
            print(f"FAIL  TE selected {backend}, not FlashAttention — the fix is not "
                  "in effect. TE's own reasoning:\n" + flash_log)
            failed += 1
        if flash_peak >= unfused_peak:
            print(f"FAIL  fused peak memory {flash_peak:.0f} MiB is not below the "
                  f"unfused reference {unfused_peak:.0f} MiB")
            failed += 1

        fused, reference = torch.load(flash_path), torch.load(unfused_path)
        for key in fused:
            if torch.allclose(fused[key], reference[key], atol=ATOL, rtol=RTOL):
                print(f"ok    {key} matches the unfused reference")
            else:
                delta = (fused[key] - reference[key]).abs().max().item()
                print(f"FAIL  {key} differs from the unfused reference "
                      f"(max abs {delta:.3e})")
                failed += 1

    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true",
                        help="Assert backend selection (run on a GPU node)")
    parser.add_argument("--dpa-child", metavar="PATH", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.dpa_child:
        dpa_child(Path(args.dpa_child))
        return

    failed = check_versions()
    if args.gpu and not failed:
        failed += check_gpu()

    if failed:
        raise SystemExit(f"{failed} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
