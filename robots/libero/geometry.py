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


"""Calibrated LIBERO observations and wrist inspection poses for a planner."""

from __future__ import annotations

import io
import threading
from collections.abc import Iterable
from typing import Annotated
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import Field

from rpent.session import EnvState
from rpent.tools import ToolResult
from rpent.tools.base import tool
from toolkits.geometry.observation import GeometryObservation
from toolkits.geometry.perception import VisualGrounder


def view_geometry_state(state: EnvState, step: int) -> ToolResult:
    """Keep calibrated full-frame coordinates while sending only the two working views."""
    try:
        record = state.get(step)
    except (LookupError, ValueError) as exc:
        return ToolResult(data={"error": str(exc)})
    payload = {
        "step": record.step_idx,
        "state": record.state,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "task_language": record.extras.get("task_language"),
        "result": record.result,
        "views": [],
    }
    images = []
    for camera in ("agentview", "wrist"):
        name = next(
            (
                n
                for n in (f"{camera}_high.png", f"{camera}.png")
                if n in record.artifacts
            ),
            None,
        )
        if name is None:
            continue
        with Image.open(
            io.BytesIO(state.load_bytes(name, step=record.step_idx))
        ) as img:
            full_shape = [img.height, img.width]
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
            payload["views"].append(
                {
                    "camera": camera,
                    "image_shape": [img.height, img.width],
                    "full_image": name,
                    "full_image_shape": full_shape,
                }
            )
    return ToolResult(data=payload, images=images)


class LiberoGeometryObservations:
    """Read only RGB, metric points and calibration at the requested recorded step."""

    def __init__(self, state: EnvState):
        self.state = state
        self.world_frame = f"libero_world/{uuid4().hex}"

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        if camera not in {"agentview", "wrist"}:
            raise ValueError("LIBERO geometry cameras are agentview and wrist")
        record = self.state.get(step)
        nn = record.step_idx
        for suffix in ("_high", ""):
            rgb_name = f"{camera}{suffix}.png"
            world_name = f"{camera}_world{suffix}.npz"
            meta_name = f"{camera}_metadata{suffix}.json"
            if meta_name not in record.artifacts:
                meta_name = f"{camera}_metadata.json"
            names = [rgb_name, world_name, meta_name]
            if not set(names).issubset(record.artifacts):
                continue
            rgb = self.state.load(rgb_name, step=nn)
            meta = self.state.load(meta_name, step=nn)
            h, w = rgb.shape[:2]
            if w * meta["height"] != h * meta["width"]:
                raise ValueError("recorded calibration and RGB aspect ratios differ")
            scale = np.diag([w / meta["width"], h / meta["height"], 1.0])
            return GeometryObservation(
                camera=camera,
                step=nn,
                world_frame_id=self.world_frame,
                rgb=rgb,
                world_xyz=self.state.load(world_name, step=nn),
                intrinsics=scale @ meta["intrinsic_K"],
                camera_to_world=meta["extrinsic_cam2world"],
                up_direction=np.array([0.0, 0.0, 1.0]),
                source_refs={
                    name: str(self.state.artifact_path(name, step=nn)) for name in names
                },
            )
        raise ValueError(f"no matching calibrated RGB-D pair for {camera} at step {nn}")

    def additional_observations(
        self, primary: GeometryObservation
    ) -> Iterable[GeometryObservation]:
        """Read the other calibrated camera from this exact state record."""
        other = "wrist" if primary.camera == "agentview" else "agentview"
        if not any(
            name in self.state.get(primary.step).artifacts
            for name in (f"{other}_high.png", f"{other}.png")
        ):
            return
        yield self.get_observation(camera=other, step=primary.step)

    @tool(readonly=True)
    def placement_opening(self, geometry_artifact: str, step: int) -> ToolResult:
        """Locate the opening of a verified upright basket from its measured upper walls and recessed interior."""
        result = self.state.load(geometry_artifact, step=step)
        if not (
            result.get("agent") == "GeometryTool"
            and result.get("measurement") == "visible_region"
            and result.get("status") == "usable"
            and result.get("world_frame_id") == self.world_frame
            and step == result.get("step")
            and step == self.state.get(-1).step_idx
        ):
            return ToolResult(
                data={
                    "status": "insufficient_evidence",
                    "reason": "current verified recipient needed",
                }
            )
        points = self.state.load(result["pointcloud_artifact"], step=step)
        center = _opening_center(np.asarray(points, dtype=float))
        if center is None:
            return ToolResult(
                data={
                    "status": "insufficient_evidence",
                    "reason": "surrounding lip and recessed interior not resolved",
                }
            )
        return ToolResult(
            data={
                "status": "usable",
                "world_xyz": center.tolist(),
                "step": step,
                "camera": result["camera"],
            }
        )


def _opening_center(points):
    """Estimate a drop centre only when a raised perimeter surrounds a measurable recess."""
    if len(points) < 100 or not np.isfinite(points).all():
        return None
    low, high = np.quantile(points[:, 2], [0.02, 0.98])
    height = high - low
    if height < 0.015:
        return None
    upper = points[points[:, 2] >= high - 0.08 * height]
    limits = np.quantile(upper[:, :2], [0.015, 0.985], axis=0)
    xy = limits.mean(axis=0)
    span = limits[1] - limits[0]
    if min(span) < 0.025:
        return None
    scaled = (upper[:, :2] - xy) / (0.5 * span)
    radius = np.linalg.norm(scaled, axis=1)
    ring = scaled[radius > 0.72]
    sectors = np.histogram(
        np.arctan2(ring[:, 1], ring[:, 0]), bins=np.linspace(-np.pi, np.pi, 9)
    )[0]
    if min(sectors) < 3 or np.mean(radius < 0.55) > 0.05:
        return None
    inner = np.max(abs((points[:, :2] - xy) / (0.5 * span)), axis=1) < 0.80
    if np.count_nonzero(inner & (points[:, 2] < high - 0.18 * height)) < 4:
        return None
    return np.r_[xy, high]


class WristInspection:
    """Plan an open-gripper observation pose using measured camera rays."""

    def __init__(
        self, observations: LiberoGeometryObservations, grounder: VisualGrounder
    ):
        self.observations = observations
        self.grounder = grounder

    @tool(readonly=True)
    def inspection_pose(
        self, query: str, attempt: Annotated[int, Field(ge=0, le=3)] = 0
    ) -> ToolResult:
        """Suggest a wrist inspection pose after geometry requests more evidence. Move there with open fingers, then call geometry on the new wrist frame."""
        state = self.observations.state
        step = state.get(-1).step_idx
        scene = self.observations.get_observation(camera="agentview", step=step)
        image_key = next(name for name in scene.source_refs if name.endswith(".png"))
        point = self.grounder.ground(state.load_bytes(image_key, step=step), query)
        h, w = scene.rgb.shape[:2]
        search = [(0.50, 0.61), (0.35, 0.63), (0.67, 0.64), (0.52, 0.75)]
        xy = np.array(search[attempt]) * [w, h]
        if point.found and point.point_xy is not None and point.image_size == (w, h):
            xy = np.asarray(point.point_xy, dtype=float)
        if not np.isfinite(xy).all() or not (0 <= xy[0] < w and 0 <= xy[1] < h):
            return ToolResult(
                data={"error": "inspection pixel outside the calibrated image"}
            )
        x, y = np.clip(np.rint(xy).astype(int), [0, 0], [w - 1, h - 1])
        patch = np.s_[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
        region = scene.world_xyz[patch][scene.valid_mask[patch]]
        if not len(region):
            return ToolResult(data={"error": "inspection region has no finite depth"})
        return self.pose_for_anchor(np.median(region, axis=0), step, attempt)

    def pose_for_anchor(self, aim, step, attempt):
        """Aim at an inspection proposal. A subsequent measurement must confirm it."""
        state = self.observations.state
        hand = self.observations.get_observation(camera="wrist", step=step)
        u, v, distance = [
            (0.50, 0.39, 0.31),
            (0.44, 0.40, 0.24),
            (0.62, 0.43, 0.35),
            (0.45, 0.57, 0.28),
        ][attempt]
        hh, hw = hand.rgb.shape[:2]
        direction = hand.camera_to_world[:3, :3] @ np.linalg.solve(
            hand.intrinsics, [u * hw, v * hh, 1.0]
        )
        if direction[2] > -0.3:
            return ToolResult(
                data={
                    "error": "wrist camera must face the workspace for this inspection"
                }
            )
        eef = np.asarray(state.get(step).state["robot0_eef_pos"])
        camera_offset = hand.camera_to_world[:3, 3] - eef
        distance = max(distance, (0.14 + camera_offset[2]) / -direction[2])
        pose = aim - distance * direction - camera_offset
        return ToolResult(
            data={
                "inspection_xyz": pose.tolist(),
                "source_step": step,
                "camera": "wrist",
            }
        )


def planner_measurement(data):
    """Send actionable evidence to the planner. The saved artifact retains the full fit."""
    keys = (
        "status",
        "query",
        "measurement",
        "step",
        "camera",
        "world_frame_id",
        "units",
        "definition",
        "evidence_status",
        "geometry_artifact",
        "reason",
        "warnings",
        "code",
        "error",
    )
    out = {k: data[k] for k in keys if data.get(k) is not None}
    if data.get("status") == "usable" and data.get("world_xyz") is not None:
        for key in ("world_xyz", "pixel_rc", "visible_bounds_xyz"):
            if key in data:
                out[key] = data[key]
    if data.get("status") == "sampled":
        for key in (
            "points_rc",
            "samples_xyz",
            "box_rc",
            "surface_xyz",
            "visible_bounds_xyz",
            "n_valid",
            "mask_artifact",
        ):
            if key in data:
                out[key] = data[key]
    elif data.get("status") == "usable" and data.get("mask_artifact"):
        out["mask_artifact"] = data["mask_artifact"]
    for name, field in data.get("geometry", {}).items():
        summary = {
            k: field[k]
            for k in (
                "unit",
                "definition",
                "fit_accepted",
                "evidence_status",
                "source_step",
                "source_camera",
                "warnings",
                "reason",
            )
            if field.get(k) is not None
        }
        summary.setdefault("source_camera", data.get("camera"))
        if field.get("fit_accepted") and data.get("status") in {
            "usable",
            "partial",
            "warning",
        }:
            summary["value"] = field.get("value")
        out.setdefault("geometry", {})[name] = summary
    views = data.get("view_evidence")
    if views:
        out["views_checked"] = [a["camera"] for a in views["attempts"]]
        if views["skipped"]:
            out["views_skipped"] = [
                {k: item[k] for k in ("camera", "reason") if k in item}
                for item in views["skipped"]
            ]
    if data.get("status") in {"partial", "warning", "insufficient_evidence"}:
        fields = out.get("geometry", {})
        missing = list(
            dict.fromkeys(
                [
                    *data.get("missing_geometry", ()),
                    *(name for name, field in fields.items() if "value" not in field),
                ]
            )
        )
        if missing:
            out["missing_geometry"] = missing
        if fields and not any("value" in field for field in fields.values()):
            out.update(
                status="insufficient_evidence",
                reason=data.get("reason") or "no_requested_fit_passed_quality_checks",
            )
    return out


class PlannerGeometry:
    """Cache read-only measurements and publish the evidence the planner can act on."""

    def __init__(self, geometry, state: EnvState):
        self.measure = geometry.geometry
        self.state = state
        self._measure_lock = threading.Lock()
        self._cache_step = None
        self._cache = {}

    def reset(self):
        with self._measure_lock:
            self._cache.clear()
            self._cache_step = None

    def geometry(
        self,
        query: str,
        camera: str,
        step: int = -1,
        measurement: str = "surface_point",
        required_geometry: list[str] | None = None,
    ) -> ToolResult:
        """Measure or retrieve the same semantic request at a pinned step."""
        with self._measure_lock:
            try:
                step = self.state.get(step).step_idx
            except (ValueError, LookupError) as exc:
                return ToolResult(
                    data={
                        "status": "error",
                        "code": "observation_unavailable",
                        "error": str(exc),
                    }
                )
            if self._cache_step != step:
                self._cache.clear()
                self._cache_step = step
            key = (
                camera,
                " ".join(query.casefold().split()),
                measurement,
                tuple(sorted(required_geometry or ())),
            )
            data = None
            if key in self._cache:
                try:
                    artifact = self._cache[key]
                    data = {
                        **self.state.load(artifact, step=step),
                        "geometry_artifact": artifact,
                    }
                except (OSError, LookupError, ValueError):
                    self._cache.pop(key)
            if data is None:
                data = self.measure(
                    query=query,
                    camera=camera,
                    step=step,
                    measurement=measurement,
                    required_geometry=required_geometry,
                ).data
                if (
                    data.get("step") == step
                    and data.get("status")
                    in {"usable", "insufficient_evidence", "warning", "partial"}
                    and data.get("geometry_artifact")
                ):
                    self._cache[key] = data["geometry_artifact"]
        return ToolResult(data=planner_measurement(data))
