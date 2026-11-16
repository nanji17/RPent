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

"""Segmentation and metric coordinate reads for explicit observation snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rpent.tools import ToolResult
from toolkits.geometry.observation import GeometryObservation

if TYPE_CHECKING:
    from rpent.robots.components.molmo_client import MolmoResult
    from rpent.robots.components.sam3_client import Sam3Result


class VisualGrounder(Protocol):
    """Select an original-image column/row point from RGB and a description."""

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Return a visible point or an absent target."""
        ...


class Segmenter(Protocol):
    """Produce masks using text or a positive row/column point prompt."""

    def segment(
        self,
        image: bytes,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Sam3Result:
        """Return a mask at the supplied RGB resolution, or no candidate."""
        ...


def back_project(observation: GeometryObservation, row: int, col: int) -> ToolResult:
    """Read an aligned metric point without origin, height or depth-range heuristics."""
    height, width = observation.rgb.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return ToolResult(data={"error": "selected pixel is outside the observation"})
    if not observation.valid_mask[row, col]:
        return ToolResult(data={"error": "selected pixel has no valid 3-D observation"})
    return ToolResult(
        data={
            "camera": observation.camera,
            "step": observation.step,
            "world_frame_id": observation.world_frame_id,
            "units": "m",
            "pixel_rc": [row, col],
            "world_xyz": observation.world_xyz[row, col].tolist(),
        }
    )
