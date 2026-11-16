# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score additional directions and dimensions against independently supplied shape references."""

from __future__ import annotations

import numpy as np


def box_order(corners, center, right, towards):
    rel = np.asarray(corners) - center
    indices = np.argsort(rel @ towards)
    far, near = indices[:2], indices[2:]
    return np.asarray(corners)[
        np.r_[far[np.argsort(rel[far] @ right)], near[np.argsort(rel[near] @ right)]]
    ]


def reference_values(
    profile, pose, camera, up, *, support_height_m=None, resting=False
):
    """Apply evaluator-only model landmarks and FK after geometry has produced predictions."""
    R, pos = pose[:3, :3], pose[:3, 3]
    center = pos + R @ np.asarray(profile["local_center"])
    right, toward = camera[:3, 0], -camera[:3, 2]
    truth = {}
    if profile.get("radius") is not None:
        n = R[:, 2]
        u = right - (right @ n) * n
        if np.linalg.norm(u) > 1e-7:
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            if v @ toward < 0:
                v = -v
            rad = profile["radius"]
            truth["inner_rim_radius"] = rad
            truth["rim_extrema"] = np.array(
                [center - rad * u, center + rad * u, center + rad * v, center - rad * v]
            )
    if profile.get("corners") is not None:
        local = np.asarray(profile["corners"])
        corners = local @ R.T + pos
        truth["top_corners"] = box_order(corners, center, right, toward)
        truth["top_size"] = np.sort(
            [np.linalg.norm(local[0] - local[1]), np.linalg.norm(local[0] - local[2])]
        )[::-1]
    if profile.get("hull") is not None:
        heights = (np.asarray(profile["hull"]) @ R.T + pos) @ up
        truth["object_height"] = float(np.ptp(heights))
        truth["supported_height"] = truth["object_height"]
        if "top_corners" in truth:
            # The reference lower corners come from the independent CAD hull's bottom.
            upper = truth["top_corners"]
            lower = upper - (upper @ up - heights.min())[:, None] * up
            truth["cuboid_corners"] = np.vstack((upper, lower))
        if support_height_m is not None:
            truth["height_above_support"] = float(heights.max() - support_height_m)
        elif resting:
            truth["height_above_support"] = truth["object_height"]
    return truth


def error_mm(name, predicted, reference):
    """Require all ordered points or all side lengths to satisfy their own threshold."""
    if predicted is None:
        return None
    pred, ref = np.asarray(predicted), np.asarray(reference)
    if pred.shape != ref.shape or not np.isfinite(pred).all():
        raise ValueError(f"invalid {name} prediction")
    e = pred - ref
    if name in ("rim_extrema", "top_corners", "cuboid_corners"):
        return float(np.max(np.linalg.norm(e, axis=-1)) * 1000)
    if name in (
        "inner_rim_radius",
        "top_size",
        "object_height",
        "supported_height",
        "height_above_support",
    ):
        return float(np.max(abs(e)) * 1000)
    return float(np.linalg.norm(e) * 1000)
