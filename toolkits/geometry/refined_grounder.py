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


"""Refine one native model point in a crop centered on its own visual prediction."""

import io

import numpy as np
from PIL import Image

from rpent.robots.components.molmo_client import MolmoResult


class RefinedGrounder:
    """A second observation of the same RGB frame, with pixels mapped back to the original raster."""

    def __init__(self, backend, fraction=0.50):
        if not 0 < fraction < 1:
            raise ValueError("crop fraction must be between zero and one")
        self.backend = backend
        self.fraction = fraction

    def ground(self, image, query):
        first = self.backend.ground(image, query)
        return self.refine(image, query, first)[0]

    def confirm(self, image, query, initial):
        return self.refine(image, query, initial)[0]

    def refine(self, image, query, first):
        raster = Image.open(io.BytesIO(image)).convert("RGB")
        width, height = raster.size
        if first.image_size != raster.size:
            raise ValueError("initial point raster differs from refinement raster")
        if not first.found or first.point_xy is None:
            return first, {"initial_xy": None, "crop": None}
        x, y = first.point_xy
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("initial point is outside the refinement raster")
        crop_w = max(1, round(width * self.fraction))
        crop_h = max(1, round(height * self.fraction))
        left = int(np.clip(round(x - crop_w / 2), 0, width - crop_w))
        top = int(np.clip(round(y - crop_h / 2), 0, height - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        buf = io.BytesIO()
        raster.crop(box).save(buf, format="PNG")
        second = self.backend.ground(buf.getvalue(), query)
        trace = {
            "initial_xy": [x, y],
            "initial_answer": first.answer,
            "crop": box,
            "crop_xy": second.point_xy,
            "crop_answer": second.answer,
        }
        if second.image_size != (crop_w, crop_h):
            raise ValueError("local crop response uses a different raster")
        if not second.found or second.point_xy is None:
            return first, trace
        sx, sy = second.point_xy
        if not (0 <= sx < crop_w and 0 <= sy < crop_h):
            raise ValueError("refined point is outside the local crop")
        return MolmoResult(
            True, (left + sx, top + sy), raster.size, second.answer
        ), trace
