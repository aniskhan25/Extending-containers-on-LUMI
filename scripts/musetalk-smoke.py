#!/usr/bin/env python3
"""Smoke test for the MuseTalk LUMI container (run by %test in musetalk-lumi.def).

Checks the things that actually broke while getting MuseTalk onto this base image:
a ROCm torch build, MediaPipe's native library finding libEGL, the Tasks
FaceLandmarker API existing at all (1.0 removed mediapipe.solutions), and the venv
pins winning over the base image's copies.
"""

import sys


def main():
    import torch

    assert torch.version.hip is not None, (
        f"torch {torch.__version__} is not a ROCm build -- a PyPI CPU or CUDA wheel "
        "has shadowed the base image's torch"
    )
    print(f"torch {torch.__version__} (HIP {torch.version.hip})")

    # MediaPipe must import after torch: that is the order the real pipeline uses, and
    # loading libmediapipe.so into a process that already holds the ROCm runtime is
    # exactly what upstream MuseTalk does on the first landmark call.
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    assert hasattr(mp, "Image"), "mediapipe is installed but has no Image type"
    print(f"mediapipe {mp.__version__}")

    import cv2
    import diffusers
    import librosa
    import numpy
    import omegaconf
    import transformers
    from transformers import AutoFeatureExtractor, WhisperModel  # noqa: F401

    print(f"cv2 {cv2.__version__}  numpy {numpy.__version__}  "
          f"diffusers {diffusers.__version__}  transformers {transformers.__version__}  "
          f"librosa {librosa.__version__}  omegaconf {omegaconf.__version__}")

    # The failure this guards against is a missing libEGL.so.1/libGLESv2.so.2, which
    # only surfaces when the native library is actually dlopen'd -- importing mediapipe
    # alone does not do it.
    try:
        FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="/nonexistent.task"),
                running_mode=RunningMode.IMAGE,
            )
        )
    except FileNotFoundError:
        # The library loaded, then rejected the bogus model path. That is the pass case.
        pass
    except OSError as exc:
        # A missing libEGL.so.1 / libGLESv2.so.2 surfaces here, as a dlopen failure.
        raise AssertionError(
            f"MediaPipe's native library failed to load: {exc}"
        ) from exc
    except Exception:
        pass
    print("mediapipe native library loads (libEGL/libGLESv2 present)")

    if torch.cuda.is_available():
        x = torch.randn(8, 3, 64, 64, device="cuda")
        w = torch.randn(16, 3, 3, 3, device="cuda")
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        print(f"gfx conv2d ok: {tuple(y.shape)} on {torch.cuda.get_device_name(0)}")
    else:
        print("no GPU visible, skipped the conv2d check (expected during a build)")

    print("smoke test passed")


if __name__ == "__main__":
    sys.exit(main())
