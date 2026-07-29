#!/usr/bin/env python3
"""Smoke test for the LUMI TransformerEngine container.

Run during build (%test): validates the ROCm torch stack, that TE's flash-attn
version gate now accepts the installed flash-attn, and that TE actually bound the
flash-attn entry points. No GPU required — TE's gate runs on import.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version as dist_version


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


def main() -> None:
    failed = check_versions()
    if failed:
        raise SystemExit(f"{failed} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
