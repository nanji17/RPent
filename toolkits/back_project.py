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

"""Read measured surfaces from a provider's calibrated image."""

from typing import Annotated

from pydantic import Field

from rpent.tools import ToolResult, tool
from toolkits.geometry.image_selection import ImageSelection, measure_selection
from toolkits.geometry.observation import ObservationProvider


@tool(readonly=True, exclude=("observations",))
def back_project(
    camera: Annotated[str, Field(min_length=1, max_length=128)],
    selection: ImageSelection,
    step: Annotated[int, Field(ge=-1)] = -1,
    *,
    observations: ObservationProvider,
) -> ToolResult:
    """Read metric surface coordinates at known pixels without model inference.

    Use the labeled full-frame image_shape [height,width] and either points_rc
    (up to 16 row,col pairs in order) or a half-open box_rc [top,left,bottom,right].
    Coordinates lie inside the declared image. The exclusive bottom and right
    may equal its height and width. For either selection, height_m [low,high]
    filters the world coordinate along the unit up axis in metres from the
    frame origin. A filtered or invalid point keeps its slot as null.
    Returned pixel indices and image_shape refer to the native image.
    The result describes sampled surfaces. Confirm the intended object in the
    current images before using its coordinates.
    """
    try:
        selection = ImageSelection.model_validate(selection)
        observation = observations.get_observation(camera=camera, step=step)
        if observation.camera != camera or (step >= 0 and observation.step != step):
            raise ValueError("Provider returned a different camera or step.")
        data = measure_selection(observation, selection)
    except (ValueError, LookupError, OSError, RuntimeError) as exc:
        return ToolResult(data={"status": "error", "error": str(exc)})
    return ToolResult(data=data)
