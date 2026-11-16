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

"""Map selections on a calibrated preview to measured surface samples."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from toolkits.geometry.observation import GeometryObservation

Index = Annotated[int, Field(strict=True, ge=0)]
Size = Annotated[int, Field(strict=True, gt=0)]


class ImageSelection(BaseModel):
    """Native or resized full-frame pixels, with exclusive bottom/right box edges."""

    model_config = ConfigDict(extra="forbid")
    image_shape: Annotated[
        list[Size],
        Field(
            min_length=2,
            max_length=2,
            description="Height and width of the image used to select these pixels.",
        ),
    ]
    points_rc: (
        Annotated[
            list[Annotated[list[Index], Field(min_length=2, max_length=2)]],
            Field(min_length=1, max_length=16),
        ]
        | None
    ) = None
    box_rc: Annotated[list[Index], Field(min_length=4, max_length=4)] | None = None
    height_m: (
        Annotated[list[FiniteFloat | None], Field(min_length=2, max_length=2)] | None
    ) = None

    @model_validator(mode="after")
    def check_extent(self):
        if (self.points_rc is None) == (self.box_rc is None):
            raise ValueError("Supply points_rc or box_rc.")
        h, w = self.image_shape
        for i, (r, c) in enumerate(self.points_rc or ()):
            if r >= h or c >= w:
                raise ValueError(
                    f"Pixel [{r}, {c}] at index {i} exceeds image_shape [{h}, {w}]. "
                    f"Choose rows 0..{h - 1} and columns 0..{w - 1} on the same "
                    "image used to declare image_shape."
                )
        if self.box_rc:
            r0, c0, r1, c1 = self.box_rc
            if not (r0 < r1 <= h and c0 < c1 <= w):
                raise ValueError(
                    f"box_rc={self.box_rc} is invalid for image_shape [{h}, {w}]. "
                    f"Use 0 <= top < bottom <= {h} and 0 <= left < right <= {w}. "
                    "Select the visible box on the same image used to declare image_shape."
                )
        if self.height_m is not None:
            low, high = self.height_m
            if low is not None and high is not None and low > high:
                raise ValueError(
                    "height_m is [low, high] in metres and requires low <= high."
                )
        return self

    def native(self, obs: GeometryObservation):
        """Map pixel centers and half-open boxes to the measured raster."""
        h, w = self.image_shape
        nh, nw = obs.rgb.shape[:2]
        if h * nw != w * nh:
            raise ValueError(
                "Selection must use the full calibrated image at its original aspect ratio."
            )
        if self.points_rc is not None:
            return [
                (
                    min(nh - 1, (2 * r + 1) * nh // (2 * h)),
                    min(nw - 1, (2 * c + 1) * nw // (2 * w)),
                )
                for r, c in self.points_rc
            ], None
        r0, c0, r1, c1 = self.box_rc
        return [], (
            r0 * nh // h,
            c0 * nw // w,
            (r1 * nh + h - 1) // h,
            (c1 * nw + w - 1) // w,
        )


def _region_summary(points: np.ndarray) -> dict:
    if len(points) < 8:
        return {
            "status": "need_observation",
            "reason": "insufficient_valid_surface_samples",
        }
    # Report a real observed point. A midpoint can land inside an empty cavity.
    center = np.median(points, axis=0)
    surface = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    return {
        "status": "sampled",
        "surface_xyz": surface.tolist(),
        "visible_bounds_xyz": [
            points.min(axis=0).tolist(),
            points.max(axis=0).tolist(),
        ],
        "n_valid": len(points),
    }


def measure_selection(obs: GeometryObservation, selection: ImageSelection) -> dict:
    """Read depth samples at the selected pixels of the pinned camera."""
    points, box = selection.native(obs)
    result: dict[str, Any] = {
        "definition": "selected_surface_only",
        "evidence_status": "identity_unchecked",
        "image_shape": list(obs.rgb.shape[:2]),
        "selection_image_shape": selection.image_shape,
        "world_frame_id": obs.world_frame_id,
        "step": obs.step,
        "camera": obs.camera,
        "units": "m",
    }
    if box is not None:
        r0, c0, r1, c1 = box
        indices = (slice(r0, r1), slice(c0, c1))
    else:
        rows, cols = np.asarray(points).T
        indices = (rows, cols)
    samples = obs.world_xyz[indices]
    valid = obs.valid_mask[indices]
    if selection.height_m is not None:
        if obs.up_direction is None:
            raise ValueError("A height interval requires the provider's up direction.")
        low, high = selection.height_m
        heights = samples[valid] @ obs.up_direction
        keep = (heights >= (-np.inf if low is None else low)) & (
            heights <= (np.inf if high is None else high)
        )
        result.update(height_m=selection.height_m, n_height_filtered=int((~keep).sum()))
        # Preserve the positions of invalid and filtered pixels in point batches.
        within_height = np.zeros_like(valid)
        within_height[valid] = keep
        valid = within_height
    if box is not None:
        result.update(_region_summary(samples[valid]), box_rc=list(box))
    else:
        result["points_rc"] = [list(p) for p in points]
        result["samples_xyz"] = [
            sample.tolist() if ok else None for sample, ok in zip(samples, valid)
        ]
        if valid.any():
            result["status"] = "sampled"
        else:
            result.update(
                status="need_observation",
                reason="no_selected_points_in_height_interval"
                if result.get("n_height_filtered")
                else "no_valid_3d_at_selected_pixels",
            )
    return result
