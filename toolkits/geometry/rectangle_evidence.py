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


"""Check known rectangular dimensions against independent observed depth components."""

import numpy as np
from scipy.ndimage import find_objects, label, maximum_filter, minimum_filter

from toolkits.geometry.registration import constrained_rectangle
from toolkits.geometry.rgbd_parts import object_points


def _agrees(quality):
    if "known_extents_m" not in quality:
        return False
    known = np.array(quality["known_extents_m"])
    measured = np.array(quality["observed_extents_m"])
    return bool(np.all(abs(measured - known) <= np.maximum(0.004, 0.18 * known)))


def reconcile(observation, candidates, regions, template, support, anchor):
    """Retain a compatible anchored face or require exactly one complete metric match."""
    cam = observation.camera_to_world
    up = observation.up_direction
    if cam is None or up is None or support is None:
        return None
    for name, cloud in candidates.items():
        if len(cloud) < 25:
            continue
        _, q = constrained_rectangle(template, cloud, up, cam[:3, 3])
        if _agrees(q):
            return None
        mask = regions[name]
        if "known_extents_m" in q and (
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        ):
            return None
    if not candidates:
        return None
    world = observation.world_xyz
    depth = (world - cam[:3, 3]) @ cam[:3, 2]
    spread = maximum_filter(depth, 3) - minimum_filter(depth, 3)
    raised = (
        observation.valid_mask & ((world @ up) > support + 0.002) & (spread < 0.012)
    )
    labels, n = label(raised)
    sizes = np.bincount(labels.ravel())
    found = []
    for i, region in enumerate(find_objects(labels, n), 1):
        if region is None or not 35 <= sizes[i] <= raised.size * 0.18:
            continue
        local_mask = labels[region] == i
        cloud = object_points(
            world[region], observation.valid_mask[region], local_mask, cam[:3, 2]
        )
        if len(cloud) < 25:
            continue
        _, q = constrained_rectangle(template, cloud, up, cam[:3, 3])
        if _agrees(q):
            found.append((i, cloud))
            if len(found) > 1:
                break
    if len(found) != 1:
        return (
            {},
            {},
            {
                "reason": "known_dimensions_not_uniquely_observed",
                "n_compatible": len(found),
            },
        )
    comp, cloud = found[0]
    mask = labels == comp
    yy, xx = np.nonzero(mask)
    gap = float(np.min(np.hypot(yy - anchor[0], xx - anchor[1])))
    return (
        {"depth": cloud},
        {"depth": mask},
        {
            "warnings": ["reidentified_by_unique_observed_face_dimensions"],
            "original_anchor_distance_px": gap,
        },
    )
