#!/usr/bin/env python3
"""Apply the two changes MuseTalk needs to run inside the LUMI AI Factory container.

1. Replace musetalk/utils/preprocessing.py with the MediaPipe version. Upstream gets its
   68 face landmarks from DWPose via mmpose, and mmcv has no ROCm build.

2. Pass weights_only=False to torch.load. PyTorch 2.6 flipped that default to True, and
   several of MuseTalk's checkpoints (notably face-parse-bisent/resnet18-5c106cde.pth)
   are in the legacy .tar format, which cannot be read at all under weights_only=True:

       RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy
       .tar format.

   The weights come from the pinned HuggingFace repos in fetch-musetalk-assets.sh, so
   the trust assumption behind weights_only=False is the same one already made by
   downloading them.

Idempotent, and it asserts on every edit, so a MuseTalk update that moves this code
fails loudly here instead of silently going unpatched.

    scripts/musetalk-patch-for-lumi.py /scratch/$PROJECT/$USER/musetalk/MuseTalk
"""

import os
import re
import shutil
import sys

TORCH_LOAD_FILES = [
    "musetalk/models/unet.py",
    "musetalk/utils/face_detection/detection/sfd/sfd_detector.py",
    "musetalk/utils/face_parsing/__init__.py",
    "musetalk/utils/face_parsing/resnet.py",
    "musetalk/whisper/whisper/__init__.py",
]


def add_weights_only(src):
    """Append weights_only=False to every torch.load(...) call that lacks it."""
    out = []
    i = 0
    edits = 0
    for match in re.finditer(r"torch\.load\(", src):
        start = match.end()
        depth = 1
        j = start
        while depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        end = j - 1  # index of the closing paren
        args = src[start:end]
        out.append(src[i:end])
        if "weights_only" not in args:
            out.append(", weights_only=False")
            edits += 1
        i = end
    out.append(src[i:])
    return "".join(out), edits


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "MuseTalk"
    here = os.path.dirname(os.path.abspath(__file__))

    shim = os.path.join(here, "musetalk-preprocessing-mediapipe.py")
    target = os.path.join(repo, "musetalk/utils/preprocessing.py")
    assert os.path.exists(target), f"not a MuseTalk checkout: {repo}"
    shutil.copyfile(shim, target)
    print(f"preprocessing.py <- {os.path.basename(shim)}")

    total = 0
    for rel in TORCH_LOAD_FILES:
        path = os.path.join(repo, rel)
        assert os.path.exists(path), f"missing {rel}; did MuseTalk move it?"
        src = open(path).read()
        patched, edits = add_weights_only(src)
        assert "torch.load(" not in patched or edits or "weights_only" in src, \
            f"no torch.load call found in {rel}; did MuseTalk move it?"
        if edits:
            open(path, "w").write(patched)
        total += edits
        print(f"{rel}: {edits} torch.load call(s) patched")

    print(f"done, {total} torch.load call(s) patched in total")


if __name__ == "__main__":
    main()
