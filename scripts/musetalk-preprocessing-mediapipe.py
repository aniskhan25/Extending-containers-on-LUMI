"""Drop-in replacement for MuseTalk's musetalk/utils/preprocessing.py on LUMI.

Upstream derives its 68 face landmarks from DWPose, which it runs through mmpose:

    from mmpose.apis import inference_topdown, init_model   # at module import time
    ...
    face_land_mark = keypoints[0][23:91]                    # COCO-WholeBody face points

mmcv (the mmpose backend) has no ROCm build and its CUDA ops do not compile against
HIP, so that import can never succeed inside the LUMI AI Factory container. This file
keeps every other part of the pipeline byte-identical and swaps only the landmark
source for MediaPipe's FaceLandmarker.

Two things make the swap safe:

* COCO-WholeBody points 23:91 are in dlib-68 order, and MediaPipe's canonical face mesh
  has a well-known 68-point subset (_MP468_TO_DLIB68 below). After re-indexing, the
  downstream arithmetic on face_land_mark[28|29|30] and on min/max over the set is
  unchanged from upstream.
* The S3FD face detector (`fa`) that produces the actual bounding box is untouched --
  MediaPipe only supplies the landmarks that trim the box down to the lower half of
  the face.

MediaPipe 1.0 removed the legacy `mediapipe.solutions` API, so this uses the Tasks API,
which needs the `face_landmarker.task` bundle on disk. scripts/fetch-musetalk-assets.sh
puts it in models/; override with MUSETALK_FACE_LANDMARKER if it lives elsewhere.
"""

import atexit
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# musetalk/utils/__init__.py appends this directory to sys.path, which is what makes
# the bare `face_detection` import below resolve.
from face_detection import FaceAlignment, LandmarksType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# S3FD face detector, unchanged from upstream.
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=str(device))

_LANDMARKER_PATH = os.environ.get(
    "MUSETALK_FACE_LANDMARKER", os.path.join("models", "face_landmarker.task")
)
if not os.path.exists(_LANDMARKER_PATH):
    raise FileNotFoundError(
        f"MediaPipe face landmarker bundle not found at {_LANDMARKER_PATH}. "
        "Run scripts/fetch-musetalk-assets.sh, or set MUSETALK_FACE_LANDMARKER."
    )

# Built at import time, mirroring upstream's init_model(...) call, so the process holds
# the same set of live models as the original.
_face_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_LANDMARKER_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
)

# MediaPipe's FaceLandmarker.__del__ runs during interpreter teardown, by which point
# the ctypes bindings it calls into are already None -- it prints "Exception ignored in
# __del__ ... TypeError: 'NoneType' object is not callable", and closing native handles
# that late is exactly the kind of thing that turns into a segfault at exit. Close it
# earlier, at atexit, and swallow the failure if MediaPipe's own dispatcher thread pool
# has already gone: by then the process is on its way out either way.
def _close_landmarker():
    try:
        _face_landmarker.close()
    except Exception:
        pass


atexit.register(_close_landmarker)

# MediaPipe canonical face mesh vertex ids for the 68 dlib landmarks, in dlib order:
# 0-16 jaw, 17-26 eyebrows, 27-35 nose, 36-47 eyes, 48-67 mouth. Only a few entries are
# actually load-bearing here -- 27..30 walk down the nose bridge (168 glabella, 197 and
# 5 on the bridge, 4 the nose tip), 8 is the chin (152), and the jaw line supplies the
# horizontal extent -- but the full table keeps the array interchangeable with the
# COCO-WholeBody slice it replaces.
_MP468_TO_DLIB68 = np.array([
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 395, 431, 323, 361, 454, 389,
    71, 63, 105, 66, 107,
    336, 296, 334, 293, 301,
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
], dtype=np.int32)

# Kept verbatim from upstream: inference.py compares bboxes against this value.
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _face_landmarks_68(frame):
    """dlib-68-ordered landmarks in pixel coordinates, or None if no face is found.

    Replaces upstream's inference_topdown(model, frame) + keypoints[0][23:91].
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
    return pts[_MP468_TO_DLIB68].astype(np.int32)


def _collect(img_list, upperbondrange):
    """Shared body of get_landmark_and_bbox and get_bbox_range."""
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        face_land_mark = _face_landmarks_68(np.asarray(fb)[0])

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None or face_land_mark is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29].copy()
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

            f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond),
                          np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]

    # Upstream divides by len() unguarded, which raises ZeroDivisionError instead of
    # reporting "no faces found" when detection fails on every frame.
    if average_range_minus:
        text_range = (f"Total frame:「{len(frames)}」 Manually adjust range : "
                      f"[ -{int(sum(average_range_minus) / len(average_range_minus))}~"
                      f"{int(sum(average_range_plus) / len(average_range_plus))} ] , "
                      f"the current value: {upperbondrange}")
    else:
        text_range = f"Total frame:「{len(frames)}」 no face detected in any frame"
    return coords_list, frames, text_range


def get_bbox_range(img_list, upperbondrange=0):
    _, _, text_range = _collect(img_list, upperbondrange)
    return text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    coords_list, frames, text_range = _collect(img_list, upperbondrange)
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(text_range)
    print("*************************************************************************************************************************************")
    return coords_list, frames


if __name__ == "__main__":
    img_list = sys.argv[1:]
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            print('no face')
            continue
        x1, y1, x2, y2 = bbox
        print('bbox', bbox, 'cropped shape', frame[y1:y2, x1:x2].shape)
